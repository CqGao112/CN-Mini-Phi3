deepspeed --include localhost:1,2,4,5,6,7 dpo_miniphi3.py \
    --deepspeed deepspeed_configs/ds_config.json > dpo.log