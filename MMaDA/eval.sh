SCRIPT_PATH=$(realpath "$0")
cd "$(dirname "$SCRIPT_PATH")/.."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_URL="your-api-url"

PRETRAINED="Gen-Verse/MMaDA-8B-MixCoT"
GEN_LENGTH=256
DIFF_STEP=256
BLOCK_LENGTH=128
NGPU=8

# # Default
accelerate launch --num_processes=${NGPU} --main_process_port=12345 -m lmms_eval \
    --model mmada \
    --model_args=pretrained=${PRETRAINED},gen_method=default,gen_length=${GEN_LENGTH},diff_step=${DIFF_STEP},block_length=${BLOCK_LENGTH},reasoning=True \
    --tasks mmmu_val_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "./default"

accelerate launch --num_processes=${NGPU} --main_process_port=12345 -m lmms_eval \
    --model mmada \
    --model_args=pretrained=${PRETRAINED},gen_method=remix,gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=0.8,beta_mix=0.4,js_threshold=0.1,task_name=mathvista,reasoning=True \
    --tasks mmmu_val_mmada \
    --batch_size 1 \
    --log_samples \
    --output_path "./remix"
