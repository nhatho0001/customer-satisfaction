from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService 
from typing import cast

model_deployer = MLFlowModelDeployer.get_active_model_deployer()
services = model_deployer.find_model_server(
    pipeline_name="continuous_deployment_pipeline",
    pipeline_step_name="mlflow_model_deployer_step",
    model_name="sklearn_regressor",
)
if services:
    service = cast(MLFlowDeploymentService, services[0])
    if services[0].is_running:
        print(
            f"Seldon deployment service started and reachable at:\n"
            f"    {services.prediction_url}\n"
        )
    elif services[0].is_failed:
        print(
            f"Seldon deployment service is in a failure state. "
            f"The last error message was: {services.status.last_error}"
        )
    else:
        print(f"Seldon deployment service is not running")

        # start the service
        services[0].start(timeout=100)

    # delete the service
    #model_deployer.delete_service(services[0].uuid, timeout=100, force=False)
else:
    print(f'not model')