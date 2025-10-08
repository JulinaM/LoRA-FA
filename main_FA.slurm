#!/bin/bash
#SBATCH --job-name="LoRA-FA"
#SBATCH --account=pgs0385
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128G

# USAGE: sbatch main_FA.slurm math 8 <tag> <extra_arg>
# --- Argument Handling ---
if [ -z "$1" ]; then
    echo "ERROR: No job type specified."
    echo "Usage: sbatch $0 {math|math_eval|code|code_eval|dialogue|dialogue_eval|alpaca|alpaca_eval|commonsense|commonsense_eval|nlu}"
    exit 1
fi
JOB_TYPE=$1
iters=$2 
run_type=$3  # test or base or default
extra_arg=$4  # test file for commonsense
wandb_name=lorafa-$iters-$run_type  # lorafa-32-test or lorafa-32-base or lorafa-32-default
echo "wandb_name: $wandb_name"

# --- Dynamic Logging ---
LOG_DIR="logs/lorafa"
mkdir -p $LOG_DIR 
if [ -n "$extra_arg" ]; then
    LOG_FILE="${LOG_DIR}/${iters}_${run_type}_${JOB_TYPE}_${extra_arg}_${SLURM_JOB_ID}.out"
else
    LOG_FILE="${LOG_DIR}/${iters}_${run_type}_${JOB_TYPE}_${SLURM_JOB_ID}.out"
fi

exec > "$LOG_FILE" 2>&1


# --- Load modules and activate environment ---
module purge
module load cuda/12.4.1
module load miniconda3/24.1.2-py310
source activate loraone
#backup venv in lora

# --- W&B API Key ---
export WANDB_API_KEY=""

# --- Change to project directory ---
cd /users/PGS0218/julina/projects/LoRA-FA/

echo "====== JOB INFO ======"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Type: $JOB_TYPE"
echo "Working dir : $(pwd)"
echo "Hostname : $(hostname)"
echo "Date : $(date)"
echo "Python : $(which python)"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "Git branch : $(git rev-parse --abbrev-ref HEAD)"   # <- Added branch info
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPUs detected: $(python -c 'import torch; print(torch.cuda.device_count())')"


echo "Executing command for job type: $JOB_TYPE"
case $JOB_TYPE in
  math)
      CUDA_VISIBLE_DEVICES=0 accelerate launch run_exp.py -m \
        ++dataset_name=meta_math \
        ++model.epochs=1 \
        ++model.eval_epochs=1 \
        ++model.saving=false \
        +init=gradient \
        ++init.direction=LoRA-FA \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=8 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=16 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++wandb.project=lorafa \
        ++init.iters=$iters \
        ++model.warmup_steps=0 \
        ++model.warmup_ratio=0.03 \
        ++model.weight_decay=0.0 \
        ++model.lr_scheduler_type=cosine \
        ++model.learning_rate=2e-4 \
      
      ckpt="./results/lorafa_meta_math/$wandb_name/9/"
      torchrun --nproc_per_node=2 eval_gsm8k.py --model_name="$ckpt" --wandb_name="$wandb_name"
    ;;

  math_eval)
    ckpt="./results/lorafa_meta_math/$wandb_name/9/"
    srun python eval_gsm8k.py --model_name=$ckpt --wandb_name=$wandb_name
    ;;

  code)
    accelerate launch run_exp.py  \
    -m ++dataset_name=codefeedback \
        ++model.epochs=1 \
        ++model.eval_epochs=1 \
        ++model.saving=false \
        +init=gradient \
        ++init.direction=LoRA-FA \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=8 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=16 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++wandb.project=lorafa \
        ++init.iters=$iters \
        ++model.warmup_steps=0 \
        ++model.warmup_ratio=0.03 \
        ++model.weight_decay=0.0 \
        ++model.lr_scheduler_type=cosine \
        ++model.learning_rate=2e-4 \

    torchrun --nproc_per_node=2 eval_humaneval.py  --wandb_name="$wandb_name"
    python -m human_eval.evaluate_functional_correctness eval_results/humaneval_samples/$wandb_name.jsonl
    ;;
  code_eval)
    torchrun --nproc_per_node=2 eval_humaneval.py  --wandb_name="$wandb_name"
    python -m human_eval.evaluate_functional_correctness eval_results/humaneval_samples/$wandb_name.jsonl
    ;;

  alpaca)
    accelerate launch run_exp.py  \
    -m ++dataset_name=alpaca \
        ++model.epochs=3 \
        ++model.eval_epochs=1 \
        ++model.saving=false \
        +init=gradient \
        ++init.direction=LoRA-FA \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=8 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=16 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++wandb.project=lorafa \
        ++init.iters=$iters \
        ++model.warmup_steps=0 \
        ++model.warmup_ratio=0.03 \
        ++model.weight_decay=0.0 \
        ++model.lr_scheduler_type=cosine \
        ++model.learning_rate=2e-4 \
        
    torchrun --nproc_per_node=2 eval_mmlu.py  --wandb_name="$wandb_name"
    ;;
  alpaca_eval)
    torchrun --nproc_per_node=2 eval_mmlu.py  --wandb_name="$wandb_name"
    ;;

  commonsense)
   accelerate launch run_exp.py  \
    -m ++dataset_name=commonsense_reasoning \
        ++model.epochs=1 \
        ++model.eval_epochs=1 \
        ++model.saving=false \
        ++model.real_batch_size=16 \
        +init=gradient \
        ++init.direction=LoRA-FA \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=32 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=64 \
        ++peft.lora_dropout=0.05 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++wandb.project=lorafa \
        ++init.iters=$iters \
        ++model.warmup_ratio=0 \
        ++model.warmup_steps=100 \
        ++model.weight_decay=0.0 \
        ++model.lr_scheduler_type=cosine \
        ++model.learning_rate=1e-4 \
        
    torchrun --nproc_per_node=2 eval_commonsense.py --wandb_name="$wandb_name" --dataset=$4
    ;;

    commonsense_eval)
    torchrun --nproc_per_node=2 eval_commonsense.py --wandb_name="$wandb_name" --dataset=$4
    ;;

dialogue)
  accelerate launch run_exp.py  \
  -m ++dataset_name=wizard_lm \
      ++model.epochs=1 \
      ++model.eval_epochs=1 \
      ++model.saving=false \
      +init=gradient \
      ++init.direction=LoRA-FA \
      ++init.weight="stable" \
      ++init.stable_gamma=128 \
      +peft=qv \
      ++peft.lora_r=8 \
      ++peft.use_rslora=True \
      ++peft.lora_alpha=16 \
      ++seed=9 \
      ++wandb.name=$wandb_name \
      ++wandb.project=lorafa \
      ++init.iters=$iters \
      ++model.warmup_steps=0 \
      ++model.warmup_ratio=0.03 \
      ++model.weight_decay=0.0 \
      ++model.lr_scheduler_type=cosine \
      ++model.learning_rate=2e-4 \
  torchrun --nproc_per_node=2 eval_mtbench.py  --wandb_name="$wandb_name"
  python fastchat/llm_judge/gen_judgment.py --model-list $wandb_name --judge-model gpt-4 --bench-name mt_bench --parallel 2
  ;;

nlu)
  CUDA_VISIBLE_DEVICES=0 accelerate launch run_exp.py  \
  -m ++dataset_name=$4 \
      model=t5base \
      ++model.epochs=10 \
      ++model.saving=false \
      +init=gradient \
      ++init.direction=LoRA-FA \
      +peft=all \
      ++peft.lora_r=8 \
      ++peft.use_rslora=True \
      ++peft.lora_alpha=16 \
      ++seed=9 \
      ++wandb.name=$wandb_name \
      ++wandb.project=lorafa-t5base \
      ++init.iters=$iters \
      ++model.warmup_ratio=0.03 \
      ++model.weight_decay=0.0 \
      ++model.warmup_steps=0 \
      ++model.lr_scheduler_type=cosine  \
      ++model.learning_rate=1e-4 \
  ;;

  *)
    echo "ERROR: Invalid job type '$JOB_TYPE'."
    echo "Valid options are: {math|code|alpaca|dialogue|math_eval|code_eval|commonsense|commonsense_eval|nlu}"
    exit 1
    ;;
esac

echo
echo "====== Job '$JOB_TYPE' finished successfully. ======"


