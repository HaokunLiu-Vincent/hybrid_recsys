"""MLflow Models From Code entry point: no pickled application objects."""
import mlflow
from recsys.inference import load_hybrid_recommender


class HybridPythonModel(mlflow.pyfunc.PythonModel):
    def load_context(self,context):
        self.recommender=load_hybrid_recommender(context.artifacts['release'])

    def predict(self,context,model_input,params=None):
        return self.recommender.predict(model_input)


mlflow.models.set_model(HybridPythonModel())
