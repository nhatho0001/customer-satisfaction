# customer-satisfaction
 Predicting how a customer will feel about a product before they even ordered it.

 ## The Solution
 In this Project, I give special consideration to the [MLflow integration of ZenML](https://docs.zenml.io/stacks-and-components/component-guide/model-deployers/mlflow). In particular, I utilize [MLflow](https://mlflow.org/) tracking to track my metrics and parameters, and MLflow deployment to deploy my model. I also use Streamlit to showcase how this model will be used in a real-world setting.

### Setup 
```bash
pip install -r requirements.txt
pip install zenml["server"]
zenml up
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```
### Training Pipeline
- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data and remove the unwanted columns.
- `train_model`: This step will train the model and save the model
  using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
- `evaluation`: This step will evaluate the model and save the metrics -- using
  MLflow autologging -- into the artifact store.
- `model_promoter`: This step compares the newly trained model against the previous production model, in case it performed better, the new model is promoted

```python
@pipeline
def train_pipeline(path: str , model_type: str = "lightgbm"):
    df = load_data(path_file = path)
    X_train , X_test , Y_train , Y_test = clean_data(df)
    model = train_model(X_train , X_test , Y_train , Y_test , model_type = model_type)
    r2_score , mse = Evaluation(model , X_test , Y_test)
    is_promoted = model_promoter(mse = mse)
    return model , is_promoted
```
![run_pipeline](_assets\continuous_deployment.png)
### Deployment Pipeline
- `model_loader`: The step loads the `production` model from the zenml model registry.
- `model_deployer`: This step deploys the model as a service using MLflow (if deployment criteria is met).

![deployment](_assets\continuous_deployment.png)