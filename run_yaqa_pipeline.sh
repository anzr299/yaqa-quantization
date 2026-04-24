#!/usr/bin/env bash
set -euo pipefail
# Full YAQA quantization pipeline for Llama/Qwen3 models:
# 1) Hessian collection (Sketch-B via FSDP)
# 2) Quantization
# 3) Convert to HF format
# 4) (Optional) End-to-end finetuning
# 5) Perplexity eval
# 6) Zeroshot eval
#
# Usage:
#   ./run_yaqa_pipeline.sh <MODEL_ID> <RUN_TAG> [GPU_LIST] [TD_X] [TD_Y] [DECODE_MODE]
#
# Example:
#   ./run_yaqa_pipeline.sh meta-llama/Llama-3.1-8B llama3_1_8b_2bit 0,1,2,3
#   ./run_yaqa_pipeline.sh Qwen/Qwen3-8B qwen3_8b_2bit 0,1,2,3

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <MODEL_ID> <RUN_TAG> [GPU_LIST] [TD_X] [TD_Y] [DECODE_MODE] [SKIP_SUBLAYERS]"
  exit 1
fi

MODEL_ID="$1"
RUN_TAG="$2"

# Auto-detect all available GPUs if not specified
DEFAULT_GPUS=$(nvidia-smi --list-gpus | awk '{print NR-1}' | paste -sd,)
GPU_LIST="${3:-$DEFAULT_GPUS}"
TD_X="${4:-16}"
TD_Y="${5:-16}"
DECODE_MODE="${6:-1mad}"
SKIP_SUBLAYERS="${7:-}"   # specific sublayers: "5_v,10_down"

# Count GPUs correctly by splitting on comma
IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS=${#GPU_ARRAY[@]}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect Python from active venv or fall back to known path
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PY="$VIRTUAL_ENV/bin/python"
else
  PY="/home/anazir/qtip_experiments/.env-qtip/bin/python"
fi

if [[ ! -x "$PY" ]]; then
  echo "Python env not found or not executable: $PY"
  echo "Please activate your venv: source /home/anazir/qtip_experiments/.env-qtip/bin/activate"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_LIST"

# Data dirs live on /local_ssd2 for space; logs stay local
DATA_ROOT="/local_ssd2/anazir/yaqa-quantization"
HESS_MODEL_KEY="${MODEL_ID//\//_}"
HESS_DIR="$DATA_ROOT/hess/$HESS_MODEL_KEY"
CKPT_DIR="$DATA_ROOT/checkpoint/$RUN_TAG"
HF_DIR="$DATA_ROOT/hf/$RUN_TAG"
LOG_DIR="$ROOT_DIR/log"
LOG_FILE="$LOG_DIR/$RUN_TAG.log"
mkdir -p "$HESS_DIR" "$CKPT_DIR" "$HF_DIR" "$LOG_DIR"

echo "[INFO] Model:    $MODEL_ID"
echo "[INFO] Run tag:  $RUN_TAG"
echo "[INFO] GPUs:     $CUDA_VISIBLE_DEVICES"
echo "[INFO] Num GPUs: $NUM_GPUS"
echo "[INFO] Python:   $PY"
echo "[INFO] Log:      $LOG_FILE"
echo "[INFO] TD_X:     $TD_X"
echo "[INFO] TD_Y:     $TD_Y"
echo "[INFO] Decode:   $DECODE_MODE"
echo "[INFO] Skip sublayers:  ${SKIP_SUBLAYERS:-none}"

# Optional dependency safety (no-op if already installed)
"$PY" -m pip install -q sentencepiece protobuf glog

# Detect model architecture to choose the right hessian script
MODEL_TYPE=$("$PY" -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('$MODEL_ID').model_type)")
echo "[INFO] Model type: $MODEL_TYPE"

if [[ "$MODEL_TYPE" == "qwen3" ]]; then
  HESS_MODULE="hessian_llama.get_hess_qwen3"
else
  HESS_MODULE="hessian_llama.get_hess_llama"
fi

{
  echo "===== STAGE 1: Hessian collection (Sketch-B) ====="
  if ls "$HESS_DIR"/*.pt 1>/dev/null 2>&1; then
    echo "[SKIP] Hessians for $MODEL_ID already exist in $HESS_DIR, skipping generation."
  else
    echo "[INFO] Collecting Hessians using module: $HESS_MODULE"
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=$NUM_GPUS \
      -m "$HESS_MODULE" \
      --orig_model "$MODEL_ID" \
      --save_path "$HESS_DIR" \
      --hessian_sketch B \
      --n_seqs 65536 \
      --batch_size 2 \
      --ctx_size 2048 \
      --power_iters 1 \
      --fp64_accum
  fi

  echo "===== STAGE 2: Quantization ====="
  if ls "$CKPT_DIR"/*.pt 1>/dev/null 2>&1; then
    echo "[SKIP] Quantized checkpoints already exist in $CKPT_DIR, skipping quantization."
  else
    "$PY" -m quantize_llama.quantize_finetune_llama \
      --save_path "$CKPT_DIR" \
      --codebook bitshift \
      --base_model "$MODEL_ID" \
      --hess_path "$HESS_DIR" \
      --sigma_reg 1e-2 \
      --scale_override 0.9 \
      --ft_epochs 0 \
      --td_x $TD_X \
      --td_y $TD_Y \
      --L 16 \
      --K 2 \
      --V 1 \
      --decode_mode "$DECODE_MODE" \
      --tlut_bits 0 \
      ${SKIP_SUBLAYERS:+--skip_list "$SKIP_SUBLAYERS"}
  fi

  echo "===== STAGE 3: Convert quantized model to HF ====="
  "$PY" -m quantize_llama.hfize_llama \
    --quantized_path "$CKPT_DIR" \
    --hf_output_path "$HF_DIR"

  echo "===== Model Size Report ====="
  "$PY" -c "
import os, sys, torch, json
from safetensors.torch import load_file

hf_dir = '$HF_DIR'
base_model = '$MODEL_ID'

# Total safetensors file size on disk
total_disk = sum(
    os.path.getsize(os.path.join(hf_dir, f))
    for f in os.listdir(hf_dir) if f.endswith('.safetensors')
)

# Break down by parameter type
st = load_file(os.path.join(hf_dir, 'model.safetensors'))
quant_bytes = 0
non_quant_bytes = 0
non_quant_params = 0
total_params = 0

for name, tensor in st.items():
    nbytes = tensor.nelement() * tensor.element_size()
    total_params += tensor.nelement()
    if 'trellis' in name:
        quant_bytes += nbytes
    else:
        non_quant_bytes += nbytes
        non_quant_params += tensor.nelement()

print(f'  Quantized model disk size:  {total_disk / (1 << 30):.2f} GB')
print(f'  Quantized weights (trellis): {quant_bytes / (1 << 20):.1f} MB')
print(f'  Non-quantized (embed/norm/lm_head): {non_quant_bytes / (1 << 20):.1f} MB')
print(f'  Total parameters: {total_params:,}')
print(f'  Effective bits/param (overall): {total_disk * 8 / total_params:.2f}')
"

  # echo "===== STAGE 4: End-to-end finetuning ====="
  # "$PY" -m quantize_llama.finetune_e2e_llama \
  #   --base_model "$MODEL_ID" \
  #   --hf_path "$HF_DIR" \
  #   --hf_output_path "${HF_DIR}_ft" \
  #   --devset_size 256 \
  #   --ft_lr 1e-5 \
  #   --ft_bs 2 \
  #   --ft_update_freq 2 \
  #   --ft_epochs 1 \
  #   --ft_early_stop 3 \
  #   --ft_grad_ckpt

  echo "===== STAGE 5: Evaluate zeroshot ====="
  accelerate launch --multi_gpu --num_processes $NUM_GPUS -m eval.eval_zeroshot \
    --tasks gsm8k,lambada_openai,gsm8k_cot_llama \
    --batch_size 4 \
    --manifest_model \
    --hf_path "$HF_DIR"

  echo "===== DONE ====="
} 2>&1 | tee -a "$LOG_FILE"
