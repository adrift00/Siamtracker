python -m torch.distributed.launch    --nproc_per_node=2  --master_port=2333   train.py --cfg configs/mobilenetv2_config.yaml
python grad_train.py --cfg configs/alexnet_grad_config.yaml
