export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Default 
torchrun --nproc_per_node=8 eval.py \
    --config configs/gsm8k.yaml \
    --method default \

# Remix
torchrun --nproc_per_node=8 eval.py \
    --config configs/gsm8k.yaml \
    --method remix \
    # --gen-kwargs threshold=0.8,js_threshold=0.3,beta_mix=0.5 \ 







