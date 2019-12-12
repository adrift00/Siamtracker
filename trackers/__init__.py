from trackers.siamrpn import SiamRPN
from trackers.meta_siamrpn import MetaSiamRPN

trackers={
    'SiamRPN': SiamRPN,
    'MetaSiamRPN': MetaSiamRPN
}
def get_tracker(tracker_name,*args):
    return trackers[tracker_name](*args)



