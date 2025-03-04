#! /bin/bash
export HF_ENDPOINT=https://hf-mirror.com

START_TIME=$SECONDS

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-240726

input_data_dir=/cpfs04/shared/MOE/datasets/data-SlimPajama/SlimPajama-627B-train-split.json
tokenizer=Qwen2Tokenizer
json_keys=text
output_data_dir=/cpfs04/shared/MOE/datasets/data-SlimPajama/slimpajama_megatron_bin_data
load_dir=/cpfs04/shared/MOE/checkpoints/qwen-ckpts/Qwen2-0.5B

INPUT="${input_data_dir}"

if [ $tokenizer = "Qwen2Tokenizer" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_qwen2_datasets \
  --patch-tokenizer-type Qwen2Tokenizer \
  --json-keys ${json_keys} \
  --load ${load_dir} \
  --workers 32 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

elif [ $tokenizer = "DeepSeekV2Tokenizer" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_deepseekv2_datasets \
  --patch-tokenizer-type DeepSeekV2Tokenizer \
  --json-keys ${json_keys} \
  --load ${load_dir} \
  --workers 8 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

elif [ $tokenizer = "LLamaTokenizer" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_llama_datasets \
  --patch-tokenizer-type LLamaTokenizer \
  --load ${load_dir} \
  --workers 16 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
