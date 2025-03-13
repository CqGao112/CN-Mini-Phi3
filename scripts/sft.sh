deepspeed --include localhost:1,2,4,5,6,7 sft_miniphi3.py \
    --deepspeed deepspeed_configs/ds_config.json > sft.log
