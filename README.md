The Official PyTorch implementation of [**LoRA-FA: Data-Aware Low-Rank Adaptation**]

### Content Overview
(Low-Rank Initialization via Gradient SVD using Functional Approximation)

Parameter-efficient fine-tuning (PEFT) methods such as Low-Rank Adaptation (LoRA) have become the de facto
strategy for adapting large pre-trained models to downstream tasks with minimal trainable parameters. Despite their
empirical success, existing LoRA variants are largely heuristic in nature, lacking a unified theoretical framework that
explains how to select the adaptation subspace, what rank is sufficient, and how to incorporate data and task-specific
structure.

In this work, we present a general framework for understanding and extending LoRA through the lens of function-
space approximation. We propose a global dual-objective formulation that jointly minimizes task-specific loss and
functional discrepancy with a stronger teacher model. This formulation introduces a flexible alignment term between
model outputs, internal representations, or transformed embeddings, governed by a tunable divergence function and
representation mapping. Crucially, we show that the most effective low-rank updates are not purely weight-dependent,
but arise from the expected gradient response of the model to the data distribution—yielding a natural, data-aware
criterion for subspace selection and rank determination.
Our framework unifies and explains a wide range of LoRA variants, including PiSSA, KaSA, GOAT, and LoRA-
Pro, as special cases of a broader principle: low-rank adaptation should be guided by data geometry, task supervision,
and semantic alignment. We further propose a cyclic optimization strategy that alternates between self-guided task
learning and periodic teacher realignment, mirroring human learning dynamics. Together, our contributions establish
a theoretical foundation for efficient, data-aware, and aligned low-rank adaptation—offering both conceptual clarity
and practical design principles for future PEFT research.

---
### Algorithmic Overview


---
### Quick Start

Specific config parameters:
```
model:
  bf16: true # set true if needed
  max_length: 1024 # input max length for training
  prec_reg: 1.0e-06 # adjust for pre-conditioners
  saving: false # if true, the model will merge adapters then save after training
init:
  mode: gradient
  direction: LoRA-FA
  max_length: 1024 # input max lenght using for computing full-batch gradient, recomment to be consistent with max_length in model
  scale: stable
  stable_gamma: 128 # gamma parameter in the init
  # the gradient batch size is bsz x iters
  bsz: 1 # sub-batch size per iteration for full gradient compute
  iters: 8 # total number of iterations for full gradient compute
```

To use LoRA-FA **without** pre-conditioners, please use the following slurm command.
```
python run_exp.py -m \
        ++dataset_name=meta_math \
        ++model.epochs=1 \
        ++model.eval_epochs=1 \
        +init=gradient \
        ++init.direction=LoRA-FA \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=8 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=16 \
        ++wandb.name=$runname \
        ++wandb.project=lorafa \
        ++init.iters=$iters 
```

For multi-GPU training, please use the following slurm command (2 GPUs example)
```
CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
--main_process_port $(shuf -i 10000-60000 -n 1) \
 run_exp.py -m \
        ++dataset_name=meta_math \
        ++model.epochs=1 \
        ++model.eval_epochs=1 \
        +init=gradient \
        ++init.direction=LoRA-FA \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=8 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=16 \
        ++wandb.name=$runname \
        ++wandb.project=lorafa \
        ++init.iters=$iters 
```

---
### Evaluation

#### Math 
torchrun --nproc_per_node=2 eval_gsm8k.py --wandb_name="$runname"
