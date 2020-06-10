import torch

from models import get_model
from pruning_model import prune_model
from utils.model_load import load_pretrain
from models.base_siam_model import BaseSiamModel
from configs.config import cfg
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if __name__ == '__main__':
    # # mobilenetv2
    # cfg.merge_from_file('configs/mobilenetv2_pruning.yaml')
    # pretrained_path = './snapshot/mobilenetv2_sfp_0_75_new/checkpoint_e2.pth'
    # model = get_model('PruningSiamModel')
    # model = load_pretrain(model, pretrained_path).cuda()  # load the mask
    # model = prune_model(model)  # refine the model
    # # model = load_pretrain(model,
    # #                       './snapshot/mobilenetv2_sfp_0_75_finetune/checkpoint_e1.pth').cuda().eval()
    # examplar = torch.randn(1, 3, 127, 127, device='cuda')
    # search = torch.randn(1, 3, 255, 255, device='cuda')
    # model.eval()
    # e0, e1, e2 = model.get_examplar(examplar)
    # torch.onnx.export(model,
    #                   (e0, e1, e2, search),
    #                   "pretrained_models/siamrpn_mobi_pruning_search.onnx",
    #                   verbose=True,
    #                   input_names=['e0', 'e1', 'e2', 'search'],
    #
    #                   output_names=['cls', 'loc'])
    #
    # # examplar convert
    # # torch.onnx.export(model,
    # #                   examplar,
    # #                   'pretrained_models/siamrpn_mobi_pruning_examplar.onnx',
    # #                   verbose=True,
    # #                   input_names=['examplar'],
    # #                   output_names=['e0', 'e1', 'e2'])

    # #alexnet
    # cfg.merge_from_file('configs/alexnet_config.yaml')
    # pretrained_path = './snapshot/alexnet_new/checkpoint_e50.pth'
    # model = get_model('BaseSiamModel').cuda()
    # model = load_pretrain(model, pretrained_path)  # load the mask
    # examplar = torch.randn(1, 3, 127, 127, device='cuda')
    # search = torch.randn(1, 3, 255, 255, device='cuda')
    # model.eval()
    # e0= model.get_examplar(examplar)
    # torch.onnx.export(model,
    #                   (e0,search),
    #                   "pretrained_models/siamrpn_alex_search.onnx",
    #                   verbose=True,
    #                   input_names=['e0','search'],
    #                   output_names=['cls', 'loc'])
    # # examplar convert
    # # torch.onnx.export(model,
    # #                   examplar,
    # #                   'pretrained_models/siamrpn_alex_examplar.onnx',
    # #                   verbose=True,
    # #                   input_names=['examplar'],
    # #                   output_names=['e0'])
    #

    cfg.merge_from_file('configs/mobilenetv2_pruning.yaml')
    pretrained_path = './snapshot/mobilenetv2_sfp_0_75_new/checkpoint_e2.pth'
    model = get_model('PruningSiamModel')
    model = load_pretrain(model, pretrained_path)  # load the mask
    model = prune_model(model).cuda().eval()  # refine the model
    examplar = torch.randn(50, 3, 127, 127, device='cuda')
    search = torch.randn(50, 3, 255, 255, device='cuda')
    torch.onnx.export(model, (examplar, search), 'pretrained_models/siamrpn_mobi_pruning.onnx',
                      verbose=True,
                      input_names=['examplar', 'search'],
                      output_names=['cls', 'loc'])

    # test
    # torch.onnx.export(model,
    #                   examplar,
    #                   'pretrained_models/siamrpn_mobi_pruning_examplar_test.onnx',
    #                   verbose=True,
    #                   input_names=['examplar'],
    #                   output_names=['e1'])
