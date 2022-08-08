#! /bin/bash

input_dim=3
max_outputs=2500
init_dim=32
n_mixtures=4
z_dim=16
hidden_dim=64
num_heads=4

num_workers=0

lr=1e-3
beta=1.0
epochs=800
scheduler="linear"
dataset_type=shapenet15k
log_name=gen/shapenet15k-tooth/camera-ready
shapenet_data_dir="/train/SetVae/ShapeNetCore.v2.PC15k"

# the argument to deepspeed (--include) means that we only run on GPU 1, not 0
deepspeed --include localhost:1 train.py \
  --cates tooth \
  --input_dim ${input_dim} \
  --max_outputs ${max_outputs} \
  --init_dim ${init_dim} \
  --n_mixtures ${n_mixtures} \
  --z_dim ${z_dim} \
  --z_scales 1 1 2 4 8 16 32 \
  --hidden_dim ${hidden_dim} \
  --num_heads ${num_heads} \
  --num_workers ${num_workers} \
  --kl_warmup_epochs 200 \
  --fixed_gmm \
  --train_gmm \
  --lr ${lr} \
  --beta ${beta} \
  --epochs ${epochs} \
  --dataset_type ${dataset_type} \
  --log_name ${log_name} \
  --shapenet_data_dir ${shapenet_data_dir} \
  --save_freq 5 \
  --viz_freq 1000 \
  --log_freq 10 \
  --val_freq 1000 \
  --scheduler ${scheduler} \
  --slot_att \
  --ln \
  --eval \
  --seed 42 \
  --distributed \
  --deepspeed_config batch_size.json

echo "Done"
exit 0
