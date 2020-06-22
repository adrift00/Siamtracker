from models.base_siam_model import BaseSiamModel
from models.grad_siam_model import GradSiamModel
from models.meta_siam_model import MetaSiamModel
from models.pruning_siam_model import PruningSiamModel

models = {'BaseSiamModel': BaseSiamModel,
          'MetaSiamModel': MetaSiamModel,
          'GradSiamModel': GradSiamModel,
          'PruningSiamModel': PruningSiamModel
          }


def get_model(model_name, **kwargs):
    return models[model_name](**kwargs)


