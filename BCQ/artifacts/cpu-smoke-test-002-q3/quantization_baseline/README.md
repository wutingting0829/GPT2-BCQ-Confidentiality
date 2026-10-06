# BCQ Weight-Level Quantization Baseline

> Historical failed baseline: Q=3 greedy. The accepted smoke-test baseline is
> the [Mixed Q6/Q8 activation-aware report](../../cpu-smoke-test-002-mixed-q6-q8-activation-aware/quantization_baseline/README.md).

This report measures how well BCQ represents the extracted GPT-2 block weights. It does not establish functional language-model utility.

- Matrices: 72
- Convention: `W_true[out_features, in_features]`
- Saved Q=3 baseline recomputation max difference: `0.0`

## Aggregate Results

| Q | Relative Frobenius error | Cosine similarity | SQNR (dB) | RMSE | MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.618980 | 0.785407 | 4.166 | 0.080878 | 0.060113 |
| 2 | 0.386464 | 0.925196 | 8.258 | 0.050497 | 0.033675 |
| 3 | 0.275201 | 0.964058 | 11.207 | 0.035959 | 0.020531 |
| 4 | 0.218976 | 0.977873 | 13.192 | 0.028612 | 0.013153 |

## Q=3 Results by Matrix Type

| Matrix type | Relative Frobenius error | Cosine similarity | SQNR (dB) |
|---|---:|---:|---:|
| W_FC | 0.252334 | 0.970016 | 11.960 |
| W_K | 0.279534 | 0.962806 | 11.071 |
| W_O | 0.300736 | 0.957251 | 10.436 |
| W_PROJ | 0.303346 | 0.955900 | 10.361 |
| W_Q | 0.251386 | 0.970252 | 11.993 |
| W_V | 0.249638 | 0.970718 | 12.054 |

Detailed per-matrix values are in `matrix_metrics.csv`; machine-readable aggregate values and provenance are in `summary.json`.

## Separate Functional Baseline

The weight-level metrics above do not measure language-model quality. After
writing Q=3 `W_BCQ` back into GPT-2, the functional evaluation on 128
WikiText-2 validation blocks is:

| Model | Validation loss down | PPL down | Next-token top-1 accuracy up | Accuracy retention |
|---|---:|---:|---:|---:|
| FP32 `W_true` | 3.318349 | 27.614711 | 39.30% | 100.00% |
| BCQ Q=3 greedy | 6.425920 | 617.648505 | 13.70% | 34.87% |

Here, 34.87% is **accuracy retention**, calculated as
`13.70% / 39.30%`. It is not the Q=3 token accuracy. For causal language
modeling, validation loss and perplexity are the primary functional metrics;
next-token accuracy is reported only as a secondary diagnostic.

The functional protocol and raw results are recorded in
`outputs/functional-baseline/comparison.json`.
