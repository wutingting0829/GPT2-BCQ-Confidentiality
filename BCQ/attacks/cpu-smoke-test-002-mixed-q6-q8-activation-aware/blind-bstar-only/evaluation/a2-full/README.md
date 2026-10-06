---
library_name: transformers
base_model: BCQ/attacks/cpu-smoke-test-002-mixed-q6-q8-activation-aware/blind-bstar-only/evaluation/best_attack_checkpoint
tags:
- generated_from_trainer
datasets:
- Salesforce/wikitext
model-index:
- name: a2-full
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# a2-full

This model is a fine-tuned version of [BCQ/attacks/cpu-smoke-test-002-mixed-q6-q8-activation-aware/blind-bstar-only/evaluation/best_attack_checkpoint](https://huggingface.co/BCQ/attacks/cpu-smoke-test-002-mixed-q6-q8-activation-aware/blind-bstar-only/evaluation/best_attack_checkpoint) on the Salesforce/wikitext wikitext-2-raw-v1 dataset.
It achieves the following results on the evaluation set:
- eval_loss: 11.9950
- eval_model_preparation_time: 0.0008
- eval_accuracy: 0.0018
- eval_runtime: 81.2673
- eval_samples_per_second: 1.575
- eval_steps_per_second: 1.575
- epoch: 0
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 1
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 5.16.0.dev0
- Pytorch 2.14.0+cu126
- Datasets 5.0.1
- Tokenizers 0.23.1
