# Attacker Input Package

This directory is the complete package given to the attacker.

The attacker may read:

- `public_b_star.safetensors`
- `manifest.json`
- public tensor shapes
- the published residual-binarization algorithm

The attacker must not read the parent experiment directory or any model-family
checkpoint/configuration. In particular, the following remain behind the
evaluator/owner boundary:

- the original `B`
- `alpha` and `z`
- random row keys `S` and `pi`
- the random seed or random-generator state
- `W_true` and `W_BCQ`
- GPT-2 identity, configuration, tokenizer, and pretrained weights

`B*` contains 72 matrices and only the values `-1` and `+1`. Every matrix row
uses an independently sampled basis permutation and sign vector.
