# DA-MoE Training

This directory contains the training code for dynamic adapter fusion based on
MoE-PEFT.

The training pipeline has two stages:

1. Train separate MixLoRA adapters for scientific knowledge and audience style.
2. Freeze the base model and adapters, then train AlphaNet to combine their
   hidden states with a token-level weight.

## Files

- `train_science_age_adapters.py`: trains science and age-style adapters.
- `train_child_adult_adapters.py`: trains child and adult-style adapters.
- `prepare_alphanet_training_data.py`: prepares balanced age-specific data.
- `train_alphanet.py`: trains AlphaNet to fuse two frozen adapters.

## Basic Usage

Run the following commands from the `MoE-PEFT` directory.

Prepare and train the adapters:

```bash
python DA_MoE_training/train_science_age_adapters.py --prepare_only
python DA_MoE_training/train_science_age_adapters.py \
  --base_model /path/to/base/model
```

Prepare the AlphaNet dataset:

```bash
python DA_MoE_training/prepare_alphanet_training_data.py
```

Train AlphaNet:

```bash
python DA_MoE_training/train_alphanet.py \
  --base_model_path /path/to/base/model \
  --adapter_a_path /path/to/science/adapter \
  --adapter_b_path /path/to/age/adapter \
  --train_json /path/to/train.json \
  --val_json /path/to/validation.json
```

Use `python <script> --help` to view all available options.
