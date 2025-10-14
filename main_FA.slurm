#!/bin/bash
#SBATCH --job-name="LoRA-FA"
#SBATCH --account=pgs0385
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128G

# USAGE: 
# sbatch main_FA.slurm --job_type=commonsense --iters=8 --epoch=1 --reg_alpha=1e-6 --tag=test --extra_arg ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
# sbatch main_FA.slurm --job_type=commonsense --iters=8 --epoch=1 --reg_alpha=1e-6 --tag=test --extra_arg=piqa

# --- Argument Handling ---
source ./parse_args.sh "$@"
WANDB_NAME="lorafa-${ITERS}-ep${EPOCHS}"

LOG_DIR="logs/lorafa"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_TYPE}_${ITERS}_ep${EPOCHS}_${REG_ALPHA}"
if [ -n "$TAG" ]; then
  LOG_FILE="${LOG_FILE}_${TAG}"
  WANDB_NAME="${WANDB_NAME}_${TAG}"
fi
if [ -n "$EXTRA_ARG" ]; then
  LOG_FILE="${LOG_FILE}_${EXTRA_ARG}"
fi
LOG_FILE="${LOG_FILE}_${SLURM_JOB_ID}.out"
exec > "$LOG_FILE" 2>&1

echo "====== ARGUMENTS ======"
echo "JOB_TYPE: $JOB_TYPE"
echo "WANDB_NAME: $WANDB_NAME"
echo "LOG_FILE: $LOG_FILE"
echo "ITERS: $ITERS"
echo "EPOCHS: $EPOCHS"
echo "REG_ALPHA: $REG_ALPHA"
echo "TAG: $TAG"
echo "EXTRA_ARG: $EXTRA_ARG"
echo "======================="
ep=$EPOCHS
iters=$ITERS
reg_alpha=$REG_ALPHA
extra_arg=$EXTRA_ARG
wandb_name=$WANDB_NAME

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
        ++model.epochs=$ep \
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
        ++model.real_batch_size=32 \
        ++init.reg_alpha=$reg_alpha
      
      ckpt="./results/lorafa_meta_math/$wandb_name/9/"
      torchrun --nproc_per_node=2 eval_gsm8k.py --model_name="$ckpt" --wandb_name="$wandb_name"
    ;;

  math_eval)
    ckpt="./results/lorafa_meta_math/$wandb_name/9/"
    srun python eval_gsm8k.py --model_name=$ckpt --wandb_name=$wandb_name
    ;;

  #lr = 5e-4
  code)
    accelerate launch run_exp.py  \
    -m ++dataset_name=codefeedback \
        ++model.epochs=$ep \
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
        ++model.real_batch_size=32 \
        ++init.reg_alpha=$reg_alpha


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
        ++model.epochs=$ep \
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
        ++model.real_batch_size=16 \
        ++init.reg_alpha=$reg_alpha

    torchrun --nproc_per_node=2 eval_mmlu.py  --wandb_name="$wandb_name"
    ;;
  alpaca_eval)
    torchrun --nproc_per_node=2 eval_mmlu.py  --wandb_name="$wandb_name"
    ;;

  commonsense)
   accelerate launch run_exp.py  \
    -m ++dataset_name=commonsense_reasoning \
        ++model.epochs=$ep \
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
        ++init.reg_alpha=$reg_alpha
        
    torchrun --nproc_per_node=2 eval_commonsense.py --wandb_name="$wandb_name" --dataset=$extra_arg
    ;;

    commonsense_eval)
    torchrun --nproc_per_node=2 eval_commonsense.py --wandb_name="$wandb_name" --dataset=$extra_arg
    ;;

dialogue)
  accelerate launch run_exp.py  \
  -m ++dataset_name=wizard_lm \
      ++model.epochs=$ep \
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
      ++init.reg_alpha=$reg_alpha

  torchrun --nproc_per_node=2 eval_mtbench.py  --wandb_name="$wandb_name"
  python fastchat/llm_judge/gen_judgment.py --model-list $wandb_name --judge-model gpt-4 --bench-name mt_bench --parallel 2
  ;;

# SST-2, CoLA, QNLI, and MRPC. {1 ×10−4,0.1,5 ×10−2, 5 ×10−2}.
#mrpc 0.01 --> old 0.00001 
#cola 0.1 *
#qnli 0.05
#sst2 1e-4
#weight decay 0.1

nlu)
  CUDA_VISIBLE_DEVICES=0 accelerate launch run_exp.py  \
  -m ++dataset_name=$extra_arg \
      model=t5base \
      ++model.epochs=$ep \
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
      ++model.learning_rate=0.01 \
      ++init.reg_alpha=$reg_alpha

  ;;

  *)
    echo "ERROR: Invalid job type '$JOB_TYPE'."
    echo "Valid options are: {math|code|alpaca|dialogue|math_eval|code_eval|commonsense|commonsense_eval|nlu}"
    exit 1
    ;;
esac

echo
echo "====== Job '$JOB_TYPE' finished successfully. ======"


