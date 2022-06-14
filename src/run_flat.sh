#!/bin/bash
#SBATCH --job-name=flat
#SBATCH --partition=2080ti,gpu
# #SBATCH -N 1
# #SBATCH -n 1
# #SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --output=../logs/flat.out

CBLUE_ROOT=../data/CBLUEDatasets

MODEL_TYPE=bert
MODEL_PATH=../bert-base-chinese
WORD_MODEL_PATH=../roberta-base-word-chinese-cluecorpussmall
USE_WORD="1"
RESUME="1"

if [ -z $RESUME ]; then
  resume=False
else
  resume=True
fi

SEED=2022
LABEL_NAMES=(labels)
TASK_ID=3

case ${TASK_ID} in
0)
  HEAD_TYPE=linear
  ;;
1)
  HEAD_TYPE=linear_nested
  LABEL_NAMES=(labels labels2)
  ;;
2)
  HEAD_TYPE=crf
  ;;
3)
  HEAD_TYPE=crf_nested
  LABEL_NAMES=(labels labels2)
  ;;
*)
  echo "Error ${TASK_ID}"
  exit 1
  ;;
esac

# ========= parse use_word ==========
if [ -z $USE_WORD ]; then
    use_word=False
    OUTPUT_DIR=../ckpts/${MODEL_TYPE}_${HEAD_TYPE}_${SEED}_flat  # NOTE: changed dir name
else
    use_word=True
    OUTPUT_DIR=../ckpts/${MODEL_TYPE}_${HEAD_TYPE}_${SEED}_flat  # NOTE: changed dir name
fi
# ===================================

PYTHONPATH=../.. \
python run_cmeee.py \
  --output_dir                  ${OUTPUT_DIR} \
  --report_to                   none \
  --overwrite_output_dir        true \
  \
  --do_train                    false \
  --do_eval                     false \
  --do_predict                  true \
  \
  --dataloader_pin_memory       true \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size  16 \
  --gradient_accumulation_steps 4 \
  --eval_accumulation_steps     500 \
  \
  --learning_rate               3e-5 \
  --weight_decay                3e-6 \
  --max_grad_norm               0.5 \
  --lr_scheduler_type           cosine \
  \
  --num_train_epochs            80 \
  --warmup_ratio                0.05 \
  --logging_dir                 ${OUTPUT_DIR} \
  \
  --logging_strategy            steps \
  --logging_first_step          true \
  --logging_steps               200 \
  --save_strategy               steps \
  --save_steps                  1000 \
  --evaluation_strategy         steps \
  --eval_steps                  1000 \
  \
  --save_total_limit            1 \
  --no_cuda                     false \
  --seed                        ${SEED} \
  --dataloader_num_workers      8 \
  --disable_tqdm                true \
  --load_best_model_at_end      true \
  --metric_for_best_model       f1 \
  --greater_is_better           true \
  \
  --model_type                  ${MODEL_TYPE} \
  --use_word                    ${use_word} \
  --model_path                  ${MODEL_PATH} \
  --word_model_path             ${WORD_MODEL_PATH} \
  --head_type                   ${HEAD_TYPE} \
  \
  --cblue_root                  ${CBLUE_ROOT} \
  --max_length                  512 \
  --label_names                 ${LABEL_NAMES[@]} \
  --resumed_training            ${resume} \
  --use_flat                    True

