#!/bin/bash
#SBATCH --job-name="LoRA-One"
#SBATCH --account=pgs0385
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G

# --- Argument Handling ---
if [ -z "$1" ]; then
    echo "ERROR: No job type specified."
    echo "Usage: sbatch $0 {math_train|math_eval|code_train|code_eval|chat|alpaca|alpaca_eval}"
    exit 1
fi
JOB_TYPE=$1
iters=$2 
run_type=$3  # test or base or default
wandb_name=loraone-$iters-$run_type # loraone-32 or loraone-16

# --- Dynamic Logging ---
LOG_DIR="logs/loraone"
mkdir -p $LOG_DIR 
LOG_FILE="${LOG_DIR}/${iters}_${run_type}_${JOB_TYPE}_${SLURM_JOB_ID}.out"
exec > "$LOG_FILE" 2>&1

# --- Load modules and activate environment ---
module purge
module load cuda/12.4.1
module load miniconda3/24.1.2-py310
source ~/.bash_rc
conda activate loraone

# --- W&B API Key ---
export WANDB_API_KEY="68145c2d2bdaa01a462e7a63353b61c9a49b1f54"

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


# --- Execute command ---
echo "Executing command for job type: $JOB_TYPE"
case $JOB_TYPE in
  math)
      accelerate launch run_exp.py -m \
        ++dataset_name=meta_math \
        ++model.epochs=3 \
        ++model.eval_epochs=1 \
        ++model.saving=false \
        +init=gradient \
        ++init.direction=LoRA-One \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=8 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=16 \
        ++model.learning_rate=2e-4 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++init.iters=$iters
      
      ckpt="/users/PGS0218/julina/projects/LoRA-FA/results/lorafa_meta_math/$wandb_name/9/"
      echo "Checkpoint: $ckpt"
      if [ ! -d "$ckpt" ]; then
          echo "ERROR: Checkpoint directory does not exist: $ckpt"
          exit 1
      fi
      torchrun --nproc_per_node=2 eval_gsm8k.py --model_name="$ckpt" --wandb_name="$wandb_name"
    ;;

  math_train)
    accelerate launch run_exp.py -m \
        ++dataset_name=meta_math \
        ++model.epochs=1 \
        ++model.eval_epochs=1 \
        ++model.saving=false \
        +init=gradient \
        ++init.direction=LoRA-One \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=8 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=16 \
        ++model.learning_rate=2e-4 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++init.iters=$iters
    ;;

  math_eval)
    ckpt="/users/PGS0218/julina/projects/LoRA-FA/results/lorafa_meta_math/$wandb_name/9/"
    echo "Checkpoint: $ckpt"
    if [ ! -d "$ckpt" ]; then
        echo "ERROR: Checkpoint directory does not exist: $ckpt"
        exit 1
    fi
    torchrun --nproc_per_node=2 eval_gsm8k.py --model_name="$ckpt" --wandb_name="$wandb_name"
  ;;
  
  code)
    accelerate launch run_exp.py  \
    -m ++dataset_name=codefeedback \
        ++model.epochs=3 \
        ++model.eval_epochs=1 \
        ++model.saving=false \
        +init=gradient \
        ++init.direction=LoRA-One \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=8 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=16 \
        ++model.learning_rate=2e-4 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++init.iters=$iters

    echo "wandb_name: $wandb_name"
    torchrun --nproc_per_node=2 eval_humaneval.py  --wandb_name="$wandb_name"
    python -m human_eval.evaluate_functional_correctness humaneval_samples/$wandb_name.jsonl
    ;;

  code_eval)
    echo "wandb_name: $wandb_name"
    torchrun --nproc_per_node=2 eval_humaneval.py  --wandb_name="$wandb_name"
    python -m human_eval.evaluate_functional_correctness humaneval_samples/$wandb_name.jsonl
    ;;
  
  chat)
    accelerate launch run_exp.py  \
    -m ++dataset_name=wizard_lm \
        ++model.epochs=1 \
        ++model.eval_epochs=1 \
        ++model.saving=false \
        +init=gradient \
        ++init.direction=LoRA-One \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=8 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=16 \
        ++model.learning_rate=2e-4 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++init.iters=$iters

    echo "wandb_name: $wandb_name"
    torchrun --nproc_per_node=2 eval_mtbench.py  --wandb_name="$wandb_name"
    python -m human_eval.evaluate_functional_correctness evaluation_results/chat/${wandb_name}_model_answers.jsonl
    ;;

    alpaca)
    accelerate launch run_exp.py  \
    -m ++dataset_name=alpaca \
        ++model.epochs=1 \
        ++model.eval_epochs=1 \
        ++model.saving=false \
        +init=gradient \
        ++init.direction=LoRA-One \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=8 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=16 \
        ++model.learning_rate=2e-4 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++wandb.project=lorafa \
        ++init.iters=$iters

    echo "wandb_name: $wandb_name"
    torchrun --nproc_per_node=2 eval_mmlu.py  --wandb_name="$wandb_name"
    ;;
    alpaca_eval)
    echo "wandb_name: $wandb_name"
    torchrun --nproc_per_node=2 eval_mmlu.py  --wandb_name="$wandb_name" 
    ;;
  *)
    echo "ERROR: Invalid job type '$JOB_TYPE'."
    echo "Valid options are: {math_train|code_train|chat_train|math_eval|code_eval}"
    exit 1
    ;;
esac

echo
echo "====== Job '$JOB_TYPE' finished successfully. ======"
