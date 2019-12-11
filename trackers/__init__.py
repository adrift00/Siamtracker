from trackers.siamrpn import SiamRPN
from trackers.siamrpn_with_update import SiamRPNWithUpdate
from trackers.meta_siamrpn import MetaSiamRPN

trackers={
    'SiamRPN': SiamRPN,
    'SiamRPNWithUpdate': SiamRPNWithUpdate,
    'MetaSiamRPN': MetaSiamRPN
}
def get_tracker(tracker_name,*args):
    return trackers[tracker_name](*args)



