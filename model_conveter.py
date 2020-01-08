import torch
from utils.model_load import load_pretrain
from models.model_builder import BaseSiamModel


if __name__=='__main__':
    pretrained_path='pretrained_models/siamrpn.pth'
    model=BaseSiamModel().cuda()
    model=load_pretrain(model,pretrained_path)
    examplar=torch.randn(128,3,127,127,device='cuda') 
    search=torch.randn(128,3,255,255,device='cuda')
    model.set_examplar(examplar)
    torch.onnx.export(model,search, "pretrained_models/siamrpn.onnx", verbose=True)

