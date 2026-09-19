# 90-Minute Scoped Fraud Sentinel

## Finding: starter notebook and notes column

- `starter_notebook.ipynb` is **not** in the workspace. Use [`fraud_sentinel.ipynb`](fraud_sentinel.ipynb).
- There is **no `notes` column** in `transactions.csv`. Redact **all string/object columns** for injection patterns.

## Must-ship

```text
CSVs → clean/merge/sanitize → rule features + weak labels
  → predictions_rules.json  (valid submission ASAP)
  → quick fp16 LoRA on Qwen2.5-1.5B
  → greedy JSON infer + rule fallback
  → predictions.json
```

| Minute | Work |
|---|---|
| 0–25 | Clean, merge, sanitize, conflict-aware dedupe |
| 25–40 | Rule features + weak labels + `predictions_rules.json` |
| 40–70 | SFT (oversample fraud ~30%), fp16 LoRA 1 epoch |
| 70–90 | Infer all → `predictions.json`; print parse fallback rate |

## Locked defaults

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tune: **fp16 LoRA** (`r=8`, 1 epoch) — no QLoRA/bnb
- Decode: greedy `temperature=0`; regex JSON parse; rule fallback on failure
- Confidence: `0.7 * p_fraud_token + 0.3 * rule_score` when available, else `rule_score`
- Split: `train_test_split`, `RANDOM_SEED=42`
- Stretch only: base-vs-LoRA table, injection demo
- Skip: README, requirements.txt, sklearn fraud model, lm-format-enforcer, env probe
