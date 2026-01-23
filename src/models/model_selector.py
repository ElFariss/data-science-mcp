"""Model Selector - Smart model selection based on dataset and constraints."""

from typing import Optional, Any
from src.models.registry import ModelRegistry, Modality, TaskType, ModelFamily


class ModelSelector:
    """Selects appropriate models based on dataset and user constraints."""
    
    BASIC_ML_MODELS = {
        "linear_regression", "ridge_regression", "lasso_regression", "elastic_net",
        "logistic_regression", "knn_classifier", "knn_regressor",
        "svc", "svr", "gaussian_nb", "multinomial_nb",
        "decision_tree_classifier", "decision_tree_regressor",
        "random_forest_classifier", "random_forest_regressor",
        "extra_trees_classifier", "gradient_boosting_classifier",
        "hist_gradient_boosting_classifier", "adaboost_classifier",
        "xgboost", "lightgbm", "catboost",
    }
    
    PRIORITY_MODELS = {
        Modality.TABULAR: [
            ("lightgbm", 100),
            ("xgboost", 95),
            ("catboost", 90),
            ("random_forest_classifier", 70),
            ("logistic_regression", 60),
            ("linear_regression", 60),
        ],
        Modality.TIMESERIES: [
            ("lightgbm", 90),
            ("xgboost", 85),
            ("prophet", 80),
            ("arima", 70),
            ("exponential_smoothing", 60),
        ],
        Modality.VISION: [
            ("efficientnet_b0", 100),
            ("resnet50", 90),
            ("mobilenet_v3", 85),
            ("vit_base", 80),
        ],
        Modality.NLP: [
            ("distilbert", 100),
            ("bert_base", 95),
            ("roberta", 90),
            ("tfidf_logistic", 70),
            ("tfidf_svm", 65),
        ],
    }
    
    def __init__(self, constraint_manager: Optional[Any] = None):
        """
        Initialize model selector.
        
        Args:
            constraint_manager: Optional constraint manager for filtering
        """
        self.constraint_manager = constraint_manager
        self._basic_ml_only = False
    
    def restrict_to_basic_ml(self, restrict: bool = True) -> None:
        """Enable/disable basic ML only mode."""
        self._basic_ml_only = restrict
    
    def select_models(
        self,
        modality: str,
        task_type: str,
        n_samples: int = 0,
        n_features: int = 0,
        has_gpu: bool = False,
        max_models: int = 5,
    ) -> list[str]:
        """
        Select appropriate models for the task.
        
        Args:
            modality: Data modality (tabular, timeseries, vision, nlp)
            task_type: Task type (classification, regression, etc.)
            n_samples: Number of samples in dataset
            n_features: Number of features
            has_gpu: Whether GPU is available
            max_models: Maximum number of models to return
            
        Returns:
            List of model names ordered by priority
        """
        try:
            mod = Modality(modality.lower())
        except ValueError:
            mod = Modality.TABULAR
        
        # Get priority models for modality
        priority_list = self.PRIORITY_MODELS.get(mod, [])
        
        selected = []
        for model_name, _ in priority_list:
            if len(selected) >= max_models:
                break
            
            spec = ModelRegistry.get(model_name)
            if spec is None:
                continue
            
            # Apply constraints
            if self._basic_ml_only and model_name not in self.BASIC_ML_MODELS:
                continue
            
            if spec.requires_gpu and not has_gpu:
                continue
            
            # Check task type compatibility
            try:
                tt = TaskType(task_type.lower().replace("binary_", "").replace("multiclass_", "multiclass"))
            except ValueError:
                tt = TaskType.CLASSIFICATION
            
            if tt not in spec.task_types:
                # Try to find a compatible variant
                continue
            
            selected.append(model_name)
        
        # If basic ML only and we don't have enough, add more basic models
        if self._basic_ml_only and len(selected) < max_models:
            for model_name in self.BASIC_ML_MODELS:
                if model_name in selected:
                    continue
                if len(selected) >= max_models:
                    break
                
                spec = ModelRegistry.get(model_name)
                if spec and mod in spec.modalities:
                    selected.append(model_name)
        
        return selected
    
    def select_baseline(self, modality: str, task_type: str) -> str:
        """
        Select a simple baseline model.
        
        Args:
            modality: Data modality
            task_type: Task type
            
        Returns:
            Model name for baseline
        """
        is_regression = "regression" in task_type.lower()
        
        if modality == "tabular":
            return "linear_regression" if is_regression else "logistic_regression"
        elif modality == "timeseries":
            return "exponential_smoothing"
        elif modality == "vision":
            return "mobilenet_v3"
        elif modality == "nlp":
            return "tfidf_logistic"
        
        return "logistic_regression"
    
    def select_primary(self, modality: str, task_type: str, has_gpu: bool = False) -> str:
        """
        Select the recommended primary model.
        
        Args:
            modality: Data modality
            task_type: Task type
            has_gpu: Whether GPU is available
            
        Returns:
            Model name for primary model
        """
        if self._basic_ml_only:
            if modality == "tabular":
                return "lightgbm"
            elif modality == "timeseries":
                return "lightgbm"
            else:
                return "random_forest_classifier"
        
        if modality == "tabular":
            return "lightgbm"
        elif modality == "timeseries":
            return "lightgbm"
        elif modality == "vision":
            if has_gpu:
                return "efficientnet_b0"
            return "mobilenet_v3"
        elif modality == "nlp":
            if has_gpu:
                return "distilbert"
            return "tfidf_logistic"
        
        return "lightgbm"
    
    def get_alternatives(
        self,
        tried_models: list[str],
        modality: str,
        max_alternatives: int = 3,
    ) -> list[str]:
        """
        Get alternative models to try.
        
        Args:
            tried_models: Models already tried
            modality: Data modality
            max_alternatives: Maximum alternatives to return
            
        Returns:
            List of alternative model names
        """
        try:
            mod = Modality(modality.lower())
        except ValueError:
            mod = Modality.TABULAR
        
        # Get all models for modality
        all_models = ModelRegistry.list_models(modality=mod)
        
        # Filter out tried models and apply constraints
        alternatives = []
        for model_name in all_models:
            if model_name in tried_models:
                continue
            if len(alternatives) >= max_alternatives:
                break
            
            if self._basic_ml_only and model_name not in self.BASIC_ML_MODELS:
                continue
            
            spec = ModelRegistry.get(model_name)
            if spec and not spec.requires_gpu:
                alternatives.append(model_name)
        
        return alternatives
