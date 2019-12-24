import itertools
import numpy as np
from colorama import Style, Fore
from toolkit.utils.statistics import calculate_accuracy, calculate_failures



class AccuracyRobustnessBenchmark(object):
    def __init__(self, dataset):
        super(AccuracyRobustnessBenchmark, self).__init__()
        self.dataset = dataset

    def eval(self, trackers):
        if isinstance(trackers, str):
            trackers = [trackers]
        result = {}
        for tracker in trackers:
            overlaps, failures = self._calc_accuracy_robustness(tracker)
            result[tracker] = {
                'overlaps': overlaps,
                'failures': failures
            }
        return result

    def show_result(self,result,show_video_level=False,helight_threshold=0.5):
        tracker_name_len = max((max([len(x) for x in result.keys()]) + 2), 12)
        header = "|{:^" + str(tracker_name_len) + "}|{:^10}|{:^12}|{:^13}|"
        header = header.format('Tracker Name',
                               'Accuracy', 'Robustness', 'Lost Number')
        formatter = "|{:^" + str(tracker_name_len) + "}|{:^10.3f}|{:^12.3f}|{:^13.1f}|"
        bar = '-' * len(header)
        print(bar)
        print(header)
        print(bar)
        tracker_names = list(result.keys())
        for tracker_name in tracker_names:
            # for tracker_name, ret in result.items():
            ret = result[tracker_name]
            overlaps = list(itertools.chain(*ret['overlaps'].values()))
            accuracy = np.nanmean(overlaps)
            length = sum([len(x) for x in ret['overlaps'].values()])
            failures = list(ret['failures'].values())
            lost_number = np.mean(np.sum(failures, axis=0))
            robustness = np.mean(np.sum(np.array(failures), axis=0) / length) * 100
            print(formatter.format(tracker_name, accuracy, robustness, lost_number))
        print(bar)

        if show_video_level and len(result) < 10:
            print('\n\n')
            header1 = "|{:^14}|".format("Tracker name")
            header2 = "|{:^14}|".format("Video name")
            for tracker_name in result.keys():
                header1 += ("{:^17}|").format(tracker_name)
                header2 += "{:^8}|{:^8}|".format("Acc", "LN")
            print('-' * len(header1))
            print(header1)
            print('-' * len(header1))
            print(header2)
            print('-' * len(header1))
            videos = list(result[tracker_name]['overlaps'].keys())
            for video in videos:
                row = "|{:^14}|".format(video)
                for tracker_name in result.keys():
                    overlaps = result[tracker_name]['overlaps'][video]
                    accuracy = np.nanmean(overlaps)
                    failures = result[tracker_name]['failures'][video]
                    lost_number = np.mean(failures)

                    accuracy_str = "{:^8.3f}".format(accuracy)
                    if accuracy < helight_threshold:
                        row += f'{Fore.RED}{accuracy_str}{Style.RESET_ALL}|'
                    else:
                        row += accuracy_str + '|'
                    lost_num_str = "{:^8.3f}".format(lost_number)
                    if lost_number > 0:
                        row += f'{Fore.RED}{lost_num_str}{Style.RESET_ALL}|'
                    else:
                        row += lost_num_str + '|'
                print(row)
            print('-' * len(header1))

    def _calc_accuracy_robustness(self, tracker):
        overlaps = {}
        failures = {}
        for video in self.dataset:
            gt_rects = video.gt_rects  #
            pred_bboxes = video.load_tracker_result(tracker)  # x,y,w,h
            failure = calculate_failures(pred_bboxes)[0]
            overlap = calculate_accuracy(pred_bboxes, gt_rects, burnin=10, bound=(video.width, video.height))[1]
            overlaps[video.name] = overlap
            failures[video.name] = failure
        return overlaps, failures




