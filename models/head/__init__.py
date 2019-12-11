from models.head.rpn import DepthwiseRPN,MultiRPN



RPNS={'DepthwiseRPN':DepthwiseRPN,
      'MultiRPN':MultiRPN}


def get_rpn_head(name,**kwargs):
    return RPNS[name](**kwargs)


