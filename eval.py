import os
import argparse
from glob import glob
from dataset.vot_dataset import VOTDataset
from configs.config import cfg
from toolkit.benchmark.ar_benchmark import AccuracyRobustnessBenchmark
from toolkit.benchmark.eao_benchmark import EAOBenchmark

parse=argparse.ArgumentParser(description='eval trackers')
parse.add_argument('--tracker',default='',type=str,help='which tracker to eval')
parse.add_argument('--dataset',default='',type=str,help='which dataset to eval')
parse.add_argument('--show_video_level',default=False,type=bool,help='whether to show detail of every video')

args=parse.parse_args()


def eval(dataset,trackers):
    ar_benchmark=AccuracyRobustnessBenchmark(dataset)
    ar_result=ar_benchmark.eval(trackers)
    ar_benchmark.show_result(ar_result,show_video_level=args.show_video_level)
    eao_benchmark=EAOBenchmark(vot_dataset)
    result=eao_benchmark.eval(trackers)
    eao_benchmark.show_result(result)



if __name__ =='__main__':
    if args.dataset=='VOT2016':
        trackers=glob(os.path.join(cfg.TRACK.RESULT_DIR,args.dataset,args.tracker+'*'))
        data_dir=os.path.join(cfg.TRACK.DATA_DIR,args.dataset)
        anno_file='VOT2016.json'
        vot_dataset=VOTDataset(data_dir,anno_file)
        eval(vot_dataset,trackers)
    elif args.dataset=='VOT2018':
        trackers=glob(os.path.join(cfg.TRACK.RESULT_DIR,args.dataset,args.tracker+'*'))
        data_dir=os.path.join(cfg.TRACK.DATA_DIR,args.dataset)
        anno_file='VOT2018.json'
        vot_dataset=VOTDataset(data_dir,anno_file)
        eval(vot_dataset,trackers)
