#!/bin/bash
# Lightweight argument parser for SLURM jobs
# Usage: source parse_args.sh "$@"

# --- Default values ---
JOB_TYPE=""
ITERS=8
EPOCHS=1
REG_ALPHA=1e-6
TAG=""
EXTRA_ARG=""

while [ $# -gt 0 ]; do
  case "$1" in
    --job_type=*|--job_type)
      JOB_TYPE="${1#*=}"; [ "$JOB_TYPE" = "$1" ] && { JOB_TYPE="$2"; shift; }
      ;;
    --iters=*|--iters)
      ITERS="${1#*=}"; [ "$ITERS" = "$1" ] && { ITERS="$2"; shift; }
      ;;
    --epoch=*|--epochs=*|--epoch|--epochs)
      EPOCHS="${1#*=}"; [ "$EPOCHS" = "$1" ] && { EPOCHS="$2"; shift; }
      ;;
    --reg_alpha=*|--reg_alpha)
      REG_ALPHA="${1#*=}"; [ "$REG_ALPHA" = "$1" ] && { REG_ALPHA="$2"; shift; }
      ;;
    --tag=*|--tag)
      TAG="${1#*=}"; [ "$TAG" = "$1" ] && { TAG="$2"; shift; }
      ;;
    --extra_arg=*|--extra_arg)
      EXTRA_ARG="${1#*=}"; [ "$EXTRA_ARG" = "$1" ] && { EXTRA_ARG="$2"; shift; }
      ;;
    -*)
      echo "❌ Unknown option: $1"
      exit 1
      ;;
    *)
      break
      ;;
  esac
  shift
done

# --- Validation ---
if [ -z "$JOB_TYPE" ]; then
  echo "❌ ERROR: --job_type is required"
  echo "Usage: sbatch main_FA.slurm --job_type=math --iters=8 --epoch=1 --reg_alpha=1e-6 --tag=test [--extra_arg=piqa]"
  exit 1
fi
