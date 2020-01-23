import torch
from models.backbone.mobilenet_v2 import MobileNetV2
     

def check_keys(model, pretrained_state_dict):
    ckpt_keys=set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())

    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        print('[Warning] missing keys: {}'.format(missing_keys))
        print('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        print('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        print('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    print('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True
 

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
        map_location=lambda storage, loc: storage.cuda(device))

    pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    try:
        check_keys(model, pretrained_dict)
    except:
        print('[Warning]: using pretrain as features.\
                Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__ =='__main__':
    pretrained_path='../mobilenetv2/snapshot/model_best.pth.tar'
    model=MobileNetV2(width_mult=1.4)
    model=load_pretrain(model,pretrained_path)
    save_path='./pretrained_models/mobilenetv2_1_4.pth'


    torch.save(model.state_dict(),save_path)



