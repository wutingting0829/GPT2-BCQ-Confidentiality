# Mixed Q6/Q8 Activation-Aware Quantization Baseline

## Decision

```text
W_true
  |
  v
Mixed Q6/Q8 + activation-aware BCQ
  |
  v
W_BCQ
  |
  v
Screening: 94.74% accuracy retention, PPL ratio = 1.1868
  |
  v
Quantization Baseline PASS
```

The screening decision above used 16 held-out WikiText-2 validation blocks.
The subsequent 128-block confirmation also passed:

| Model | Validation loss down | PPL down | Token accuracy up | Accuracy retention |
|---|---:|---:|---:|---:|
| FP32 `W_true` | 3.318349 | 27.614711 | 39.30% | 100.00% |
| Mixed Q6/Q8 activation-aware `W_BCQ` | 3.500232 | 33.123143 | 37.10% | 94.41% |

- Full-evaluation PPL ratio: `1.199475`
- Provisional gate: accuracy retention at least 90% and PPL ratio at most 1.2
- Full-evaluation decision: **PASS**, close to the PPL boundary

Validation loss and PPL are the primary causal-language-modeling metrics.
Next-token top-1 accuracy and accuracy retention are secondary diagnostics.

## BCQ Configuration

- Matrices: all 72 GPT-2 attention and MLP matrices
- Base Q: 6
- Sensitive Q: 8
- Q8 rule: all matrices in layer 0, plus `W_O` and `W_PROJ` in layers 1-11
- Matrix counts: 44 at Q6 and 28 at Q8
- Effective parameter-weighted Q: 6.930556
- Optimizer: alternating binary updates and joint per-row least squares
- Maximum alternating iterations: 50
- Activation objective: diagonal input second-moment weighted reconstruction error
- Calibration data: 16 WikiText-2 train blocks, each 1024 tokens
- Functional evaluation data: WikiText-2 validation split
- Matrix convention: `W_true[out_features, in_features]`

The activation-aware objective is

```text
sum_j E[x_j^2] * (W_true[i,j] - W_BCQ[i,j])^2
```

It prioritizes weight errors on input dimensions that are active more often.
Consequently, raw Frobenius error can be worse while functional PPL is better.

## Weight-Level Verification

| Metric | Result |
|---|---:|
| Relative Frobenius error | 0.112769 |
| Cosine similarity | 0.993621 |
| SQNR | 18.956 dB |
| Stored reconstruction max difference | 0.0 |

The public artifact contains only binary matrices with values `-1` and
`+1`. The secret artifact contains per-row `alpha` and `z`. Reconstructing all
72 matrices from `B`, `alpha`, and `z` exactly matches the saved `W_BCQ`.

## Reproducibility

- Transformers commit: `7c65cdb570646e8b01cc30d069579f2f4b60c398`
- Quantization manifest: `../manifest.json`
- Screening report: `BCQ/sweeps/mixed-q6-q8-activation-aware-50/activation_aware_results.json`
- Full `W_true` metrics: `outputs/functional-baseline/wtrue/eval_results.json`
- Full `W_BCQ` metrics: `outputs/functional-baseline/wbcq-mixed-q6-q8-activation-aware/eval_results.json`
- Loadable checkpoint: `outputs/cpu-smoke-test-002-bcq-mixed-q6-q8-activation-aware`
- Machine-readable decision: `summary.json`

This is still a smoke-test baseline because the source checkpoint was
fine-tuned for only 10 steps. The same gate must be rerun on the final
fine-tuned checkpoint before using the result as a paper claim.
