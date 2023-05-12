#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export NAME=LSTM-LoS-2

export DATA_PATH=YOUR_PATH
export TRAIN_BATCH_SIZE=512
export VAL_BATCH_SIZE=512
export TEST_BATCH_SIZE=512
export NUM_WORKERS=10
# 512, 32, 32
# 20

export LR=5e-5
export WEIGHT_DECAY=0.0

export TASK_NAME=LoS
export LOSS_TYPE=msle
export HIDDEN_SIZE=128
export N_LAYERS=2
export DIAGNOSIS_SIZE=64
export LAST_LINEAR_SIZE=17
export ALPHA=100
export DROPOUT_RATE=0.2
export MOMENTUM=0.1

export ES_MONITOR=val_loss
export ES_MIN_DELTA=0.0
export ES_PATIENCE=5
export ES_MODE=min

export OUTPUT_DIR=YOUR_PATH
#export TB_SAVE_DIR=YOUR_PATH
#export CKPT_DIRPATH=YOUR_PATH
export CKPT_SAVE_TOP_K=2
export CKPT_MONITOR=val_loss
export CKPT_MODE=min

export WANDB_KEY=
export WANDB_PROJECT=LoS-workspace-2z

export GPUS=-1
export MAX_EPOCHS=8
export LOG_EVERY_STEP=1000
export VAL_EVERY_STEP=1000
export PRECISION=16 # 32

wandb login "$WANDB_KEY" --relogin
python run.py \
  --name="$NAME" \
  --data_path="$DATA_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --val_batch_size=$VAL_BATCH_SIZE \
  --test_batch_size=$TEST_BATCH_SIZE \
  --num_workers=$NUM_WORKERS \
  --learning_rate=$LR \
  --weight_decay=$WEIGHT_DECAY \
  --task_name="$TASK_NAME" \
  --loss_type="$LOSS_TYPE" \
  --hidden_size=$HIDDEN_SIZE \
  --n_layers=$N_LAYERS \
  --diagnosis_size=$DIAGNOSIS_SIZE \
  --last_linear_size=$LAST_LINEAR_SIZE \
  --alpha=$ALPHA \
  --dropout_rate=$DROPOUT_RATE \
  --momentum=$MOMENTUM \
  --es_monitor="$ES_MONITOR" \
  --es_min_delta=$ES_MIN_DELTA \
  --es_patience=$ES_PATIENCE \
  --es_mode=$ES_MODE \
  --ckpt_save_top_k=$CKPT_SAVE_TOP_K \
  --ckpt_monitor="$CKPT_MONITOR" \
  --ckpt_mode=$CKPT_MODE \
  --wandb_key="$WANDB_KEY" \
  --wandb_project="$WANDB_PROJECT" \
  --gpus=$GPUS \
  --max_epochs=$MAX_EPOCHS \
  --log_every_n_steps=$LOG_EVERY_STEP \
  --val_check_interval=$VAL_EVERY_STEP \
  --ckpt \
  --ckpt_verbose \
  --es \
  --es_verbose \
  --tb

#--labs_only
#--no_labs
