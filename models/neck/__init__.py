from models.neck.neck import AdjustAllLayer,AdjustLayer




NECKS={'AdjustAllLayer':AdjustAllLayer,
       'AdjustLayer':AdjustLayer}



def get_neck(name,**kwargs):
    return NECKS[name](**kwargs)