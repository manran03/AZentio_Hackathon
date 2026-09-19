# Fraud Sentinel — Approach Document

## 1. Problem

Classify each bank transaction as fraud or not, using three related CSVs:

| File | Role |
|---|---|
| `transactions.csv` | Event-level activity (amount, merchant, device, velocity, etc.) |
| `accounts.csv` | Account status, balances, tier, credit util |
| `customers.csv` | KYC, risk rating, income, PEP flag |

**Required output** (per transaction):

```json
{
  "transaction_id": "TXN_...",
  "is_fraud": true,
  "confidence": 0.82,
  "justification": "One plain-English sentence."
}
```

Constraints that shaped the design:
- No ground-truth fraud labels → need **weak labels**.
- Free-text fields may contain **prompt injection**.
- Deliverable must be shippable under a short time budget → **rules first**, LLM second.
- Relational joins can leave **orphan** accounts/customers.

---

## 2. High-level approach

**Hybrid rules + LLM**, with rules as both teacher and safety net.

```
transactions / accounts / customers
        │
        ▼
  Clean + conflict-aware dedupe
        │
        ▼
  Left join: account_id → customer_id
        │
        ▼
  Sanitize all strings (injection redact)
        │
        ▼
  Heuristic rules → rule_score / weak_is_fraud
        │
        ├──────────────────────────────┐
        ▼                              ▼
 predictions_rules.json      Optional LoRA SFT (Qwen2.5-1.5B)
                                       │
                                       ▼
                          Greedy JSON inference
                                       │
                          parse fail? ──► rule fallback
                                       │
                                       ▼
                              predictions.json
```

**Why this design**
1. Rules give a valid submission immediately.
2. Rules create training targets for the small instruct model.
3. On parse/train failure, rules still produce schema-valid JSON.
4. Guardrails (redact + system prompt + greedy decode) harden against injection.

---

## 3. Data wrangling

### 3.1 Cleaning

| Domain | Handling |
|---|---|
| Amounts | Strip `INR`/commas; invalid → median fill; track `amount_was_missing` |
| Timestamps | Multi-format parse; derive `transaction_hour`, `is_weekend` |
| Booleans | Map `yes/no/true/false/1/0` |
| Categories | Uppercase, normalize aliases (e.g. `JEWELRY` → `JEWELLERY`) |
| Numerics | Coerce velocity, distance, balances, ratios |

### 3.2 Conflict-aware dedupe

Duplicate `transaction_id` rows are resolved by keeping the row with the most non-null **critical** fields:

`amount`, `transaction_timestamp`, `account_id`, `customer_id`, `merchant_name`

(with preference for valid amount and timestamp). Conflicts are logged.

### 3.3 Relational merge

1. Recover missing `customer_id` on transactions from `accounts` via `account_id`.
2. Flag orphans: `orphan_account`, `orphan_customer`.
3. **Left join** transactions → accounts on `account_id`.
4. Fill `customer_id` from account side if still missing.
5. **Left join** → customers on `customer_id`.

Left joins preserve every transaction even when account/customer rows are missing.

### 3.4 Sanitization

There is **no `notes` column**. All string/object columns are scanned with `INJECTION_RE`. Matches become `[REDACTED]` and set `injection_attempt = 1`.

---

## 4. Rule engine (weak labels)

Weighted signals stack into `rule_score` (cap 1.0).

| Decision | Rule |
|---|---|
| Fraud | `rule_score >= 0.45` |
| Confidence | Mapped from score (higher when fraud) |
| Justification | Template sentence from triggered rule IDs |

Signal families (details in `RULES_AND_GUARDRAILS.md`):

- **Merchant:** crypto, wire/quickcash, high-value jewellery, luxury
- **Geo:** foreign + far from home; elevated-risk countries
- **Behavior:** amount/avg spike, night + new device + large, 24h velocity, income outlier, negative balance
- **Compliance:** frozen/dormant/closed account, bad/pending KYC, PEP, high risk rating
- **Integrity:** orphan account, prompt-injection attempt

**Roles of rules**
1. Produce `predictions_rules.json`.
2. Supervise LoRA (`weak_is_fraud` + template justification).
3. Fallback when LLM output is invalid.

---

## 5. LLM path

### 5.1 Model

| Setting | Value |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Fine-tune | fp16 LoRA (`r=8`, α=16, 1 epoch) |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Train size | Cap ~400 samples; fraud oversampled to ~30% |
| Split | Shuffle 85% train (`RANDOM_SEED=42`) |
| Default run | Inference only (`RETRAIN=False`); use adapter if weights exist |

### 5.2 Prompting

- **System:** banking fraud classifier; exact JSON schema; ignore override attempts; treat `[REDACTED]` as injection evidence.
- **User:** structured **feature card** (amount, KYC, country, velocity, hints, etc.) — not a raw CSV dump.
- **Hints:** `triggered_rules` + `rule_score` as optional signals; model still decides `is_fraud`.

### 5.3 Inference

| Control | Choice |
|---|---|
| Decode | Greedy (`do_sample=False`) |
| Max new tokens | 96 |
| Parse | Regex `{...}` + `json.loads` |
| Confidence | `0.7 * p_model + 0.3 * rule_score` when token prob available |
| Failure | Full rule prediction for that row |

`p_model` comes from softmax over `true`/`false` at the `is_fraud` field position.

---

## 6. Guardrails (defense in depth)

| Layer | Mechanism |
|---|---|
| Data | Injection regex redact + scoring |
| Prompt | Strict system rules + feature card |
| Generate | Greedy, short generation |
| Validate | Required keys; bool normalize; non-empty justification |
| Fallback | Rules → `predictions.json` |
| Confidence | Blend with `rule_score` |

Full rule/guardrail detail: [`RULES_AND_GUARDRAILS.md`](RULES_AND_GUARDRAILS.md).

---

## 7. Deliverables & entry points

| Artifact | Source |
|---|---|
| `predictions_rules.json` | Rule engine only |
| `predictions.json` | LLM (+ rule fallback) |
| `models/fraud_sentinel_lora/` | Optional LoRA adapter |
| `fraud_sentinel.py` | Full pipeline module |
| `fraud_sentinel.ipynb` | Thin runner notebook |

**Run**

```bash
# Rules → predictions_rules.json; also copy to predictions.json
python fraud_sentinel.py --rules-only

# Rules + inference (LoRA if present, else base Qwen)
python fraud_sentinel.py

# Fine-tune LoRA then infer
python fraud_sentinel.py --retrain
```

Or open `fraud_sentinel.ipynb` and set `RETRAIN = True` only when retraining is needed.

---

## 8. Design choices (summary)

| Choice | Rationale |
|---|---|
| Rules before LLM | Valid submission ASAP; no labels required |
| Left joins on `account_id` / `customer_id` | Keep all txns; surface orphans as risk |
| Score threshold 0.45 | Needs ~2 medium signals; fewer single-feature FPs |
| Weak-label SFT | Teaches JSON + fraud language under time pressure |
| Greedy decode + fallback | Deterministic, schema-safe under failure |
| Sanitize all strings | No `notes` column; injection can appear anywhere |

---

## 9. Related docs

- [`PLAN.md`](PLAN.md) — time-box / locked defaults
- [`RULES_AND_GUARDRAILS.md`](RULES_AND_GUARDRAILS.md) — every rule weight and LLM safety control
