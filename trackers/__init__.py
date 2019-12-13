from trackers.siamrpn import SiamRPN
from trackers.meta_siamrpn import MetaSiamRPN
from trackers.graph_siamrpn import GraphSiamRPN

trackers={
    'SiamRPN': SiamRPN,
    'MetaSiamRPN': MetaSiamRPN,
    'GraphSiamRPN': GraphSiamRPN
}
def get_tracker(tracker_name,*args):
    return trackers[tracker_name](*args)



