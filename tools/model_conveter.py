import torch

from models import get_model
from pruning_model import prune_model
from utils.model_load import load_pretrain
from models.base_siam_model import BaseSiamModel
from configs.config import cfg
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if __name__ == '__main__':
    # cfg.merge_from_file('configs/mobilenetv2_finetune.yaml')
    # pretrained_path = 'pretrained_models/siamrpn_mobi.pth'
    # model = BaseSiamModel().cuda()
    # model = load_pretrain(model, pretrained_path)
    # examplar = torch.randn(1, 3, 127, 127, device='cuda')
    # search = torch.randn(1, 3, 255, 255, device='cuda')
    # model.eval()
    # e0,e1,e2 = model.get_examplar(examplar)
    # torch.onnx.export(model,
    #                   (e0,e1,e2, search),
    #                   "pretrained_models/siamrpn_mobi_search.onnx",
    #                   verbose=True,
    #                   input_names=['e0', 'e1','e2','search'],
    #                   output_names=['cls','loc'])
    # examplar convert
    # torch.onnx.export(model,examplar,'pretrained_models/siamrpn_mobi_examplar.onnx',verbose=True,output_names=['e0','e1','e2'])

    # cfg.merge_from_file('configs/mobilenetv2_finetune.yaml')
    # pretrained_path = 'pretrained_models/siamrpn_mobi.pth'
    # model = BaseSiamModel().cuda()
    # model = load_pretrain(model, pretrained_path)
    # examplar = torch.randn(1, 3, 127, 127, device='cuda')
    # search = torch.randn(1, 3, 255, 255, device='cuda')
    # model.eval()
    # torch.onnx.export(model,
    #                   (examplar, search),
    #                   "pretrained_models/siamrpn_mobi.onnx",
    #                   verbose=True,
    #                   input_names=['examplar','search'],
    #                   output_names=['cls','loc'])

    cfg.merge_from_file('configs/mobilenetv2_finetune.yaml')
    pretrained_path = './snapshot/mobilenetv2_sfp_0_75/checkpoint_e12.pth'
    model = get_model('PruningSiamModel')
    model = load_pretrain(model, pretrained_path)  # load the mask
    model = prune_model(model)  # refine the model
    model = load_pretrain(model,
                          './snapshot/mobilenetv2_sfp_0_75_finetune/checkpoint_e20.pth').cuda().eval()  # load the finetune weight
    examplar = torch.randn(1, 3, 127, 127, device='cuda')
    search = torch.randn(1, 3, 255, 255, device='cuda')
    model.eval()
    # e0, e1, e2 = model.get_examplar(examplar)
    # torch.onnx.export(model,
    #                   (e0, e1, e2, search),
    #                   "pretrained_models/siamrpn_mobi_pruning_search.onnx",
    #                   verbose=True,
    #                   input_names=['e0', 'e1', 'e2', 'search'],

    #                   output_names=['cls', 'loc'])
    # examplar convert
    torch.onnx.export(model,examplar,'pretrained_models/siamrpn_mobi_pruning_examplar.onnx',verbose=True,output_names=['e0','e1','e2'])
