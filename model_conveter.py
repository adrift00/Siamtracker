import torch
from utils.model_load import load_pretrain
from models.model_builder import BaseSiamModel

if __name__ == '__main__':
    pretrained_path = 'pretrained_models/siamrpn.pth'
    model = BaseSiamModel().cuda()
    model = load_pretrain(model, pretrained_path)
    examplar = torch.randn(1, 3, 127, 127, device='cuda')
    search = torch.randn(1, 3, 287, 287, device='cuda')
    # examplar = model.set_examplar(examplar)
    # torch.onnx.export(model,
    #                   (examplar, search),
    #                   "pretrained_models/siamrpn_search.onnx",
    #                   verbose=True,
    #                   input_names=['examplar', 'search'],
    #                   output_names=['cls','loc'])
    torch.onnx.export(model,examplar,'pretrained_models/siamrpn_examplar.onnx',verbose=True)
