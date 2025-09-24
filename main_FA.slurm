#!/bin/bash
#SBATCH --job-name="LoRA-FA"
#SBATCH --account=pgs0385
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128G

# USAGE: sbatch main_FA.slurm math_train test
# --- Argument Handling ---
if [ -z "$1" ]; then
    echo "ERROR: No job type specified."
    echo "Usage: sbatch $0 {compare|math|math_train|math_eval|code|code_train|code_eval|dialogue|chat_train|chat_eval|alpaca|alpaca_eval|commonsense}"
    exit 1
fi
JOB_TYPE=$1
iters=$2 
run_type=$3  # test or base or default
wandb_name=lorafa-$iters-$run_type  # lorafa-32-test or lorafa-32-base or lorafa-32-default

# --- Dynamic Logging ---
LOG_DIR="logs/lorafa"
mkdir -p $LOG_DIR # Create logs directory if it doesn't exist
LOG_FILE="${LOG_DIR}/${iters}_${run_type}_${JOB_TYPE}_${SLURM_JOB_ID}.out"
exec > "$LOG_FILE" 2>&1


# --- Load modules and activate environment ---
module purge
module load cuda/12.4.1
module load miniconda3/24.1.2-py310
source activate loraone

# --- W&B API Key ---
export WANDB_API_KEY="68145c2d2bdaa01a462e7a63353b61c9a49b1f54"

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


echo "Executing command for job type: $JOB_TYPE"
case $JOB_TYPE in
  compare)
    accelerate launch compare.py  \
    -m ++dataset_name=meta_math \
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
       ++model.learning_rate=2e-4 \
       ++seed=9 \
       ++wandb.name=$wandb_name \
       ++wandb.project=lorafa \
       ++init.iters=$iters
    ;;

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
        ++model.learning_rate=2e-4 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++wandb.project=lorafa \
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
    accelerate launch run_exp.py  \
    -m ++dataset_name=meta_math \
       ++model.epochs=1 \
       ++model.eval_epochs=1 \
       ++model.saving=true \
       +init=gradient \
       ++init.direction=LoRA-FA \
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
        ++model.learning_rate=2e-4 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++wandb.project=lorafa \
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
        ++model.learning_rate=2e-4 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++wandb.project=lorafa \
        ++init.iters=$iters
    echo "wandb_name: $wandb_name"
    torchrun --nproc_per_node=2 eval_mtbench.py  --wandb_name="$wandb_name"
    python -m human_eval.evaluate_functional_correctness evaluation_results/dialogue/${wandb_name}_model_answers.jsonl
    python fastchat/llm_judge/gen_judgment.py --model-list $wandb_name --judge-model gpt-4 --bench-name mt_bench --parallel 2
    ;;
  alpaca)
    accelerate launch run_exp.py  \
    -m ++dataset_name=alpaca \
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

  commonsense)
   accelerate launch run_exp.py  \
    -m ++dataset_name=commonsense_reasoning \
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
        ++model.learning_rate=2e-5 \
        ++seed=9 \
        ++wandb.name=$wandb_name \
        ++wandb.project=lorafa \
        ++init.iters=$iters

    echo "wandb_name: $wandb_name"
    torchrun --nproc_per_node=2 eval_commonsense.py --wandb_name="$wandb_name" --dataset=$4
    ;;
  *)
    echo "ERROR: Invalid job type '$JOB_TYPE'."
    echo "Valid options are: {math_train|code_train|chat_train|math_eval|code_eval}"
    exit 1
    ;;
esac

echo
echo "====== Job '$JOB_TYPE' finished successfully. ======"


