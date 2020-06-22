import torch



if __name__ == '__main__':
    model=torch.load('../pretrained_models/model_best.pth.tar')
    state={'model': model['state_dict']}
    torch.save(state,'../pretrained_models/mobilenetv2_1.4.pth')


