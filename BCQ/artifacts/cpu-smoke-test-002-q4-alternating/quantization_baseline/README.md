# BCQ Weight-Level Quantization Baseline

This report measures how well BCQ represents the extracted GPT-2 block weights. It does not establish functional language-model utility.

- Matrices: 72
- Convention: `W_true[out_features, in_features]`
- Saved Q=4 baseline recomputation max difference: `0.0`

## Aggregate Results

| Q | Relative Frobenius error | Cosine similarity | SQNR (dB) | RMSE | MAE |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.176377 | 0.984323 | 15.071 | 0.023046 | 0.014093 |

## Q=4 Results by Matrix Type

| Matrix type | Relative Frobenius error | Cosine similarity | SQNR (dB) |
|---|---:|---:|---:|
| W_FC | 0.160641 | 0.987013 | 15.883 |
| W_K | 0.183681 | 0.982986 | 14.719 |
| W_O | 0.175910 | 0.984406 | 15.094 |
| W_PROJ | 0.197178 | 0.980368 | 14.103 |
| W_Q | 0.162095 | 0.986775 | 15.805 |
| W_V | 0.160778 | 0.986991 | 15.875 |

Detailed per-matrix values are in `matrix_metrics.csv`; machine-readable aggregate values and provenance are in `summary.json`.
