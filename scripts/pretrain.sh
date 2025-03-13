deepspeed --include localhost:1,2,4,5,6,7 train_deepspeed.py \
    --deepspeed deepspeed_configs/ds_config.json > ds_pretrain.log