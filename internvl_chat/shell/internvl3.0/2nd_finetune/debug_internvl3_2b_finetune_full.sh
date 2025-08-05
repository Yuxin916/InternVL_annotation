set -x

export CUDA_VISIBLE_DEVICES=0
# Sets the GPUS variable to 8 unless it is already defined in the environment
# how many GPUs will be used for training
GPUS=1
# Sets the BATCH_SIZE variable to 128 unless it is already defined in the environment
# total number of images processed at each forward pass across all GPUs
BATCH_SIZE=2
# batch size for each GPU is 4 unless specified
# local batch size per GPU
PER_DEVICE_BATCH_SIZE=1
# gradient accumulation steps. the number of iterations required to achieve the total batch size across all GPUs
# control how many forward passes (mini-batches) are performed before performing a backward pass and updating model parameters
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
echo "Total batch size: $BATCH_SIZE; Per device: $PER_DEVICE_BATCH_SIZE; GPU: $GPUS; Gradient Accumulation: $GRADIENT_ACC"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# distributed computing settings
export MASTER_PORT=34229
# TensorFlow log level to suppress most of the logs.
# 3 only allows error messages to be printed
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# log
OUTPUT_DIR='../../all_log/internvl_v3/debug_internvl3_2b_finetune_full'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 1
# batch size per gpu: 1
# gradient accumulation steps: 2
# total batch size: 2
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path '../pretrained/InternVL3-2B' \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/habitat_hdt_sample_debug.json" \
 --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "wandb" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"