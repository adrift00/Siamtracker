from trackers.siamrpn import SiamRPN
from trackers.meta_siamrpn import MetaSiamRPN
from trackers.graph_siamrpn import GraphSiamRPN
from trackers.grad_siamrpn import GradSiamRPN

trackers={
    'SiamRPN': SiamRPN,
    'MetaSiamRPN': MetaSiamRPN,
    'GraphSiamRPN': GraphSiamRPN,
    'GradSiamRPN': GradSiamRPN
}
def get_tracker(tracker_name,*args):
    return trackers[tracker_name](*args)



