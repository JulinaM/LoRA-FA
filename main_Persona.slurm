#!/bin/bash
#SBATCH --job-name="NEWS"
#SBATCH --account=pgs0385
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128G

#USAGE: 
  # sbatch main_Persona.slurm train 8 V1 mistral One lamp_news_headlines
  # sbatch main_Persona.slurm train 8 V1 llama One lamp_news_headlines
  # sbatch main_Persona.slurm train 8 V3 mistral FA lamp_news_headlines
  # sbatch main_Persona.slurm train 8 V3 llama FA lamp_news_headlines

if [ -z "$1" ]; then
    echo "ERROR: No job type specified."
    echo "Usage: sbatch $0 {train1|train|eval|infer}"
    exit 1
fi
if [ -z "$4" ]; then
    echo "ERROR: No model_type specified."
    echo "Usage: sbatch $0 {llama|mistral}"
    exit 1
fi
if [ -z "$5" ]; then
    echo "ERROR: No direction specified."
    echo "Usage: sbatch $0 {FA|One}"
    exit 1
fi

if [ -z "$6" ]; then
    echo "ERROR: No direction specified."
    echo "Usage: sbatch $0 {lamp_news_headlines|lamp_scholarly_titles|longlamp_abstract_generation|longlamp_product_review|longlamp_topic_writing}"
    exit 1
fi

JOB_TYPE=$1
iters=$2 
version=$3  # <version> test or base or default
model_type=$4  # <model_type>
direction=$5 # <direction> FA or One
task_name=$6 #task_name="lamp_news_headlines"
seed=42

lw_direction="${direction,,}"
name=lora$lw_direction
wandb_name=$name-$iters-$version-$model_type # loraone-8-V1-mistral or lorafa-8-test-llama

# --- Dynamic Logging ---
LOG_DIR="logs_persona"/$task_name/${direction}
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/${iters}_${version}_${task_name}_${JOB_TYPE}_${model_type}_${SLURM_JOB_ID}.out"
exec > "$LOG_FILE" 2>&1

# --- Load modules and activate environment ---
module purge
# module load python/3.10
module load cuda/12.4.1
module load miniconda3/24.1.2-py310
source ~/.bash_rc
conda activate loraone
export PATH=$HOME/.local/bin:$PATH
export PYTHONNOUSERSITE=1  # prevent mixing user packages
pip show torch || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


# --- W&B API Key ---
export WANDB_API_KEY=""
export HUGGING_FACE_HUB_TOKEN=""
cd /users/PGS0218/julina/projects/LoRA-FA/LongLaMP-Benchmark
export PYTHONPATH=$(pwd):$PYTHONPATH
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
echo "task_name: $task_name"
echo "wandb_name: $wandb_name"
echo "seed: $seed"

MODEL_PATH="./results/persona"_"$task_name/$wandb_name/$seed"
TEST_FILE="/users/PGS0218/julina/projects/personalization_data/$task_name/test_k1*.jsonl"
INFER_OUT="/users/PGS0218/julina/projects/personalization_data/$task_name/outputs/"
INFER_SCRIPT_VLLM="/users/PGS0218/julina/projects/personalization_data/utils/personalized_vllm.py"
# INFER_SCRIPT="eval_persona.py"
EVAL_SCRIPT="/users/PGS0218/julina/projects/personalization_data/utils/personalized_eval.py"
case $JOB_TYPE in
  train)
    accelerate launch run_exp.py  \
    -m ++dataset_name=$task_name \
        model=$model_type \
        ++model.epochs=5 \
        ++model.eval_epochs=1 \
        ++model.saving=false \
        +init=gradient \
        ++init.direction=LoRA-$direction \
        ++init.weight="stable" \
        ++init.stable_gamma=128 \
        +peft=qv \
        ++peft.lora_r=64 \
        ++peft.use_rslora=True \
        ++peft.lora_alpha=64 \
        ++seed=$seed \
        ++wandb.name=$wandb_name \
        ++wandb.project=persona \
        ++init.iters=$iters \
        ++model.warmup_steps=0 \
        ++model.warmup_ratio=0.03 \
        ++model.weight_decay=0.0 \
        ++model.lr_scheduler_type=cosine \
        ++model.learning_rate=1e-4 \

    echo "[Train] Done. Merged model at: $MODEL_PATH"
    echo "[Infer] Infering: $TEST_FILE"
    torchrun --nproc_per_node=2 eval_persona.py  \
          --base_model=$model_type \
          --wandb_name="$wandb_name" \
          --adapter_dir $MODEL_PATH \
          --data $TEST_FILE \
          --out_dir $INFER_OUT \
          --max_tokens 512

    RESULTS_JSON=$INFER_OUT"/"$wandb_name"_results.json"
    echo "[Eval] Evaluating: $RESULTS_JSON"
    torchrun --nproc_per_node=2  "$EVAL_SCRIPT"  --data "$RESULTS_JSON" --name "$wandb_name"
  ;;

  infer_vllm)
    echo "[Infer] Infering: $TEST_FILE"
    torchrun --nproc_per_node=2  eval_persona_vllm.py   \
      --data $TEST_FILE \
      --out_dir $INFER_OUT \
      --max_tokens 512

    # torchrun --nproc_per_node=2 "$INFER_SCRIPT_VLLM" \
    #   --model "$OUTPUT_PATH/merged_checkpoint" \
    #   --data $TEST_FILE \
    #   --out_dir $INFER_OUT \
    #   --max_tokens 512

    RESULTS_JSON=$INFER_OUT"/"$wandb_name"_results.json"
    echo "[Eval] Evaluating: $RESULTS_JSON"
    torchrun --nproc_per_node=2  "$EVAL_SCRIPT"  --data "$RESULTS_JSON" --name "$wandb_name"

  ;;

  infer)
    echo "[Infer] Infering: $TEST_FILE"
    torchrun --nproc_per_node=2 eval_persona.py  \
      --base_model=$model_type \
      --wandb_name="$wandb_name" \
      --adapter_dir $MODEL_PATH \
      --data $TEST_FILE \
      --out_dir $INFER_OUT \
      --max_tokens 512

    RESULTS_JSON=$INFER_OUT"/"$wandb_name"_results.json"
    echo "[Eval] Evaluating: $RESULTS_JSON"
    torchrun --nproc_per_node=2  "$EVAL_SCRIPT"  --data "$RESULTS_JSON" --name "$wandb_name"
  ;;

  eval)
    RESULTS_JSON=$INFER_OUT"/"$wandb_name"_results.json"
    echo "[Eval] Evaluating: $RESULTS_JSON"
    torchrun --nproc_per_node=2  "$EVAL_SCRIPT"  --data "$RESULTS_JSON" --name "$wandb_name"
  ;;
  
  *)
  echo "ERROR: Invalid job type '$JOB_TYPE'."
  echo "Valid options are: {train|infer|eval}"
  exit 1
  ;;
esac
echo
echo "====== Job '$JOB_TYPE' finished successfully. ======"
