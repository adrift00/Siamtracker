from trackers.siamrpn import SiamRPN
from trackers.meta_siamrpn import MetaSiamRPN
from trackers.grad_siamrpn import GradSiamRPN

trackers={
    'SiamRPN': SiamRPN,
    'MetaSiamRPN': MetaSiamRPN,
    'GradSiamRPN': GradSiamRPN
}
def get_tracker(tracker_name,*args):
    return trackers[tracker_name](*args)



