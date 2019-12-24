from .vot import VOTDataset
from .got10k import GOT10kDataset



def get_dataset(name,*args):
    if name in ['VOT2016','VOT2018']:
        return VOTDataset(*args) 
    elif name == 'GOT-10k':
        return GOT10kDataset(*args)
    else:
        raise Exception('invalid dataset name!')

