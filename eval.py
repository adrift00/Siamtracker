import os
import argparse
from glob import glob
from dataset.vot_dataset import VOTDataset
from configs.config import cfg
from toolkit.benchmark.ar_benchmark import AccuracyRobustnessBenchmark
from toolkit.benchmark.eao_benchmark import EAOBenchmark

parse=argparse.ArgumentParser(description='eval trackers')
parse.add_argument('--tracker',default='',type=str,help='which tracker to eval')
parse.add_argument('--show_video_level',default=False,type=bool,help='whether to show detail of every video')

args=parse.parse_args()


def eval():
    trackers=glob(os.path.join(cfg.TRACK.RESULT_DIR,args.tracker+'*'))
    vot_dataset=VOTDataset(cfg.TRACK.DATA_DIR,cfg.TRACK.ANNO_FILE)
    ar_benchmark=AccuracyRobustnessBenchmark(vot_dataset)
    ar_result=ar_benchmark.eval(trackers)
    ar_benchmark.show_result(ar_result,show_video_level=args.show_video_level)
    eao_benchmark=EAOBenchmark(vot_dataset)
    result=eao_benchmark.eval(trackers)
    eao_benchmark.show_result(result)



if __name__ =='__main__':
    eval()