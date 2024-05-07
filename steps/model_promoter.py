from zenml import Model, get_step_context , step
from zenml.logger import get_logger
import logging

logger = get_logger(__name__)

@step
def model_promoter(mse: float, stage: str = "production") -> bool:
    zenml_model = get_step_context().model 
    previous_production_model = Model(name= zenml_model.name , 
                                      version= "production") 
    try:
        previous_production_model_mse = float(
            previous_production_model.get_artifact("sklearn_regressor")
            .run_metadata["metrics"]
            .value["mse"]
        )
    except:
        previous_production_model_mse = mse + 100 
    
    if mse > previous_production_model_mse:
        logger.info(
            f"Model mean-squared error {mse:.2f} is higher than"
            f" the mse of the previous production model "
            f"{previous_production_model_mse:.2f} ! "
            f"Not promoting model."
        )
        is_promoted = False
    else:
        logger.info(f"Model promoted to {stage}!")
        is_promoted = True
        zenml_model = get_step_context().model
        zenml_model.set_stage(stage, force=True)

    return is_promoted