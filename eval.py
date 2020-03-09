import os
import argparse
from glob import glob
from toolkit.datasets import get_dataset
from configs.config import cfg
from toolkit.benchmark.ar_benchmark import AccuracyRobustnessBenchmark
from toolkit.benchmark.eao_benchmark import EAOBenchmark

parse = argparse.ArgumentParser(description='eval trackers')
parse.add_argument('--result_dir', default='./result', type=str, help='the dir where result stored')
parse.add_argument('--tracker', default='', type=str, help='which tracker to eval')
parse.add_argument('--dataset', default='', type=str, help='which dataset to eval')
parse.add_argument('--show_video_level', default=False, type=bool, help='whether to show detail of every video')

args = parse.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def eval(dataset, trackers):
    ar_benchmark = AccuracyRobustnessBenchmark(dataset)
    ar_result = ar_benchmark.eval(trackers)
    ar_benchmark.show_result(ar_result, show_video_level=args.show_video_level)
    eao_benchmark = EAOBenchmark(dataset)
    result = eao_benchmark.eval(trackers)
    eao_benchmark.show_result(result)


if __name__ == '__main__':
    trackers = glob(os.path.join(args.result_dir, args.dataset, args.tracker+'*'))
    data_dir = os.path.join(cfg.TRACK.DATA_DIR, args.dataset)
    dataset = get_dataset(args.dataset, data_dir)
    if args.dataset in ['VOT2016', 'VOT2018']:
        eval(dataset, trackers)
    elif args.dataset == 'GOT-10k':
        pass
    else:
        raise Exception('dataset invalid!')
