from sklearn.base import RegressorMixin
from zenml import Model, step 

@step
def model_loader(model_name: str) -> RegressorMixin:
    model = Model(name= model_name , version="production")
    model_artifact : RegressorMixin = model.load_artifact(name= "sklearn_regressor")
    return model_artifact 
