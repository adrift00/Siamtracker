python test.py --tracker SiamRPN --dataset VOT2018 --cfg configs/mobilenetv2_pruning.yaml --snapshot ./snapshot/mobilenetv2_sfp_0_75_finetune/checkpoint_e20.pth
python test.py --tracker GradSiamRPN --dataset VOT2018 --cfg configs/alexnet_grad_config.yaml --snapshot ./snapshot/grad_single_loss_multi_data/checkpoint_e7.pth 
