from models.base_siam_model import BaseSiamModel
from models.grad_siam_model import GradSiamModel
from models.graph_siam_model import GraphSiamModel
from models.meta_siam_model import MetaSiamModel

models = {'BaseSiamModel': BaseSiamModel,
          'MetaSiamModel': MetaSiamModel,
          'GraphSiamModel': GraphSiamModel,
          'GradSiamModel': GradSiamModel
          }


def get_model(model_name, **kwargs):
    return models[model_name](**kwargs)


