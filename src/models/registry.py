"""Model Registry - Central catalog of all supported models."""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Type
from enum import Enum


class Modality(str, Enum):
    TABULAR = "tabular"
    TIMESERIES = "timeseries"
    VISION = "vision"
    NLP = "nlp"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    FORECASTING = "forecasting"
    ANOMALY = "anomaly"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    NER = "ner"
    QA = "qa"


class ModelFamily(str, Enum):
    LINEAR = "linear"
    DISTANCE = "distance"
    SVM = "svm"
    NAIVE_BAYES = "naive_bayes"
    TREE = "tree"
    ENSEMBLE = "ensemble"
    BOOSTING = "boosting"
    GAM = "gam"
    NEURAL = "neural"
    FACTORIZATION = "factorization"
    AUTOML = "automl"
    ANOMALY = "anomaly"
    STATISTICAL = "statistical"
    PROPHET = "prophet"
    DEEP_LEARNING = "deep_learning"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    CLASSICAL = "classical"


@dataclass
class ModelSpec:
    """Specification for a model in the registry."""
    name: str
    family: ModelFamily
    modalities: list[Modality]
    task_types: list[TaskType]
    requires_gpu: bool = False
    training_time: str = "fast"  # fast, medium, slow
    memory_requirement: str = "low"  # low, medium, high
    sklearn_compatible: bool = True
    default_params: dict = field(default_factory=dict)
    hyperparameter_space: dict = field(default_factory=dict)
    description: str = ""
    library: str = "sklearn"


class ModelRegistry:
    """Central registry for all models."""
    
    _models: dict[str, ModelSpec] = {}
    
    @classmethod
    def register(cls, spec: ModelSpec) -> None:
        """Register a model."""
        cls._models[spec.name.lower()] = spec
    
    @classmethod
    def get(cls, name: str) -> Optional[ModelSpec]:
        """Get a model spec by name."""
        return cls._models.get(name.lower())
    
    @classmethod
    def list_models(
        cls,
        modality: Optional[Modality] = None,
        task_type: Optional[TaskType] = None,
        family: Optional[ModelFamily] = None,
        requires_gpu: Optional[bool] = None,
    ) -> list[str]:
        """List models matching criteria."""
        result = []
        for name, spec in cls._models.items():
            if modality and modality not in spec.modalities:
                continue
            if task_type and task_type not in spec.task_types:
                continue
            if family and spec.family != family:
                continue
            if requires_gpu is not None and spec.requires_gpu != requires_gpu:
                continue
            result.append(name)
        return sorted(result)
    
    @classmethod
    def get_all(cls) -> dict[str, ModelSpec]:
        """Get all registered models."""
        return cls._models.copy()


# ============================================================================
# Register Tabular Models
# ============================================================================

# Linear Models
ModelRegistry.register(ModelSpec(
    name="linear_regression",
    family=ModelFamily.LINEAR,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.REGRESSION],
    default_params={},
    description="Simple linear regression",
    library="sklearn",
))

ModelRegistry.register(ModelSpec(
    name="ridge_regression",
    family=ModelFamily.LINEAR,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.REGRESSION],
    default_params={"alpha": 1.0},
    hyperparameter_space={"alpha": (0.001, 100)},
    description="Linear regression with L2 regularization",
))

ModelRegistry.register(ModelSpec(
    name="lasso_regression",
    family=ModelFamily.LINEAR,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.REGRESSION],
    default_params={"alpha": 1.0},
    hyperparameter_space={"alpha": (0.001, 100)},
    description="Linear regression with L1 regularization",
))

ModelRegistry.register(ModelSpec(
    name="elastic_net",
    family=ModelFamily.LINEAR,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.REGRESSION],
    default_params={"alpha": 1.0, "l1_ratio": 0.5},
    description="Linear regression with L1+L2 regularization",
))

ModelRegistry.register(ModelSpec(
    name="logistic_regression",
    family=ModelFamily.LINEAR,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    default_params={"max_iter": 1000},
    description="Logistic regression classifier",
))

# Distance-Based
ModelRegistry.register(ModelSpec(
    name="knn_classifier",
    family=ModelFamily.DISTANCE,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    default_params={"n_neighbors": 5},
    hyperparameter_space={"n_neighbors": (1, 50)},
    description="K-Nearest Neighbors classifier",
))

ModelRegistry.register(ModelSpec(
    name="knn_regressor",
    family=ModelFamily.DISTANCE,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.REGRESSION],
    default_params={"n_neighbors": 5},
    description="K-Nearest Neighbors regressor",
))

# SVM
ModelRegistry.register(ModelSpec(
    name="svc",
    family=ModelFamily.SVM,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    default_params={"kernel": "rbf", "C": 1.0},
    training_time="medium",
    description="Support Vector Classifier",
))

ModelRegistry.register(ModelSpec(
    name="svr",
    family=ModelFamily.SVM,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.REGRESSION],
    default_params={"kernel": "rbf", "C": 1.0},
    training_time="medium",
    description="Support Vector Regressor",
))

# Naive Bayes
ModelRegistry.register(ModelSpec(
    name="gaussian_nb",
    family=ModelFamily.NAIVE_BAYES,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    description="Gaussian Naive Bayes",
))

ModelRegistry.register(ModelSpec(
    name="multinomial_nb",
    family=ModelFamily.NAIVE_BAYES,
    modalities=[Modality.TABULAR, Modality.NLP],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    description="Multinomial Naive Bayes (good for text)",
))

# Trees
ModelRegistry.register(ModelSpec(
    name="decision_tree_classifier",
    family=ModelFamily.TREE,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    default_params={"max_depth": None},
    description="Decision Tree Classifier",
))

ModelRegistry.register(ModelSpec(
    name="decision_tree_regressor",
    family=ModelFamily.TREE,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.REGRESSION],
    default_params={"max_depth": None},
    description="Decision Tree Regressor",
))

# Ensembles
ModelRegistry.register(ModelSpec(
    name="random_forest_classifier",
    family=ModelFamily.ENSEMBLE,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    default_params={"n_estimators": 100, "n_jobs": -1},
    hyperparameter_space={"n_estimators": (50, 500), "max_depth": (3, 20)},
    description="Random Forest Classifier",
))

ModelRegistry.register(ModelSpec(
    name="random_forest_regressor",
    family=ModelFamily.ENSEMBLE,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.REGRESSION],
    default_params={"n_estimators": 100, "n_jobs": -1},
    description="Random Forest Regressor",
))

ModelRegistry.register(ModelSpec(
    name="extra_trees_classifier",
    family=ModelFamily.ENSEMBLE,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    default_params={"n_estimators": 100, "n_jobs": -1},
    description="Extra Trees Classifier",
))

# Boosting (HIGH PRIORITY)
ModelRegistry.register(ModelSpec(
    name="xgboost",
    family=ModelFamily.BOOSTING,
    modalities=[Modality.TABULAR, Modality.TIMESERIES],
    task_types=[TaskType.CLASSIFICATION, TaskType.REGRESSION, TaskType.MULTICLASS],
    default_params={"n_estimators": 100, "learning_rate": 0.1, "n_jobs": -1},
    hyperparameter_space={
        "n_estimators": (50, 500),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "subsample": (0.6, 1.0),
    },
    training_time="medium",
    description="XGBoost - Extreme Gradient Boosting",
    library="xgboost",
))

ModelRegistry.register(ModelSpec(
    name="lightgbm",
    family=ModelFamily.BOOSTING,
    modalities=[Modality.TABULAR, Modality.TIMESERIES],
    task_types=[TaskType.CLASSIFICATION, TaskType.REGRESSION, TaskType.MULTICLASS],
    default_params={"n_estimators": 100, "learning_rate": 0.1, "n_jobs": -1},
    hyperparameter_space={
        "n_estimators": (50, 500),
        "learning_rate": (0.01, 0.3),
        "num_leaves": (20, 100),
        "subsample": (0.6, 1.0),
    },
    training_time="fast",
    description="LightGBM - Light Gradient Boosting Machine",
    library="lightgbm",
))

ModelRegistry.register(ModelSpec(
    name="catboost",
    family=ModelFamily.BOOSTING,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.REGRESSION, TaskType.MULTICLASS],
    default_params={"iterations": 100, "learning_rate": 0.1, "verbose": False},
    hyperparameter_space={
        "iterations": (50, 500),
        "learning_rate": (0.01, 0.3),
        "depth": (4, 10),
    },
    training_time="medium",
    description="CatBoost - Great for categorical features",
    library="catboost",
))

ModelRegistry.register(ModelSpec(
    name="gradient_boosting_classifier",
    family=ModelFamily.BOOSTING,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    default_params={"n_estimators": 100, "learning_rate": 0.1},
    training_time="medium",
    description="Sklearn Gradient Boosting Classifier",
))

ModelRegistry.register(ModelSpec(
    name="hist_gradient_boosting_classifier",
    family=ModelFamily.BOOSTING,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    default_params={"max_iter": 100, "learning_rate": 0.1},
    training_time="fast",
    description="Histogram-based Gradient Boosting (fast for large datasets)",
))

ModelRegistry.register(ModelSpec(
    name="adaboost_classifier",
    family=ModelFamily.BOOSTING,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    default_params={"n_estimators": 50},
    description="AdaBoost Classifier",
))

# Anomaly Detection
ModelRegistry.register(ModelSpec(
    name="isolation_forest",
    family=ModelFamily.ANOMALY,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.ANOMALY],
    default_params={"n_estimators": 100, "contamination": "auto"},
    description="Isolation Forest for anomaly detection",
))

ModelRegistry.register(ModelSpec(
    name="one_class_svm",
    family=ModelFamily.ANOMALY,
    modalities=[Modality.TABULAR],
    task_types=[TaskType.ANOMALY],
    default_params={"kernel": "rbf", "nu": 0.1},
    description="One-Class SVM for anomaly detection",
))

# ============================================================================
# Register Time Series Models
# ============================================================================

ModelRegistry.register(ModelSpec(
    name="arima",
    family=ModelFamily.STATISTICAL,
    modalities=[Modality.TIMESERIES],
    task_types=[TaskType.FORECASTING],
    sklearn_compatible=False,
    default_params={"order": (1, 1, 1)},
    description="ARIMA - Autoregressive Integrated Moving Average",
    library="statsmodels",
))

ModelRegistry.register(ModelSpec(
    name="sarima",
    family=ModelFamily.STATISTICAL,
    modalities=[Modality.TIMESERIES],
    task_types=[TaskType.FORECASTING],
    sklearn_compatible=False,
    default_params={"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
    description="Seasonal ARIMA",
    library="statsmodels",
))

ModelRegistry.register(ModelSpec(
    name="exponential_smoothing",
    family=ModelFamily.STATISTICAL,
    modalities=[Modality.TIMESERIES],
    task_types=[TaskType.FORECASTING],
    sklearn_compatible=False,
    description="Exponential Smoothing (Holt-Winters)",
    library="statsmodels",
))

ModelRegistry.register(ModelSpec(
    name="prophet",
    family=ModelFamily.PROPHET,
    modalities=[Modality.TIMESERIES],
    task_types=[TaskType.FORECASTING],
    sklearn_compatible=False,
    default_params={"yearly_seasonality": True, "weekly_seasonality": True},
    description="Facebook Prophet",
    library="prophet",
))

# ============================================================================
# Register Vision Models
# ============================================================================

ModelRegistry.register(ModelSpec(
    name="resnet50",
    family=ModelFamily.CNN,
    modalities=[Modality.VISION],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    requires_gpu=True,
    training_time="slow",
    memory_requirement="high",
    sklearn_compatible=False,
    description="ResNet-50 (Transfer Learning)",
    library="timm",
))

ModelRegistry.register(ModelSpec(
    name="efficientnet_b0",
    family=ModelFamily.CNN,
    modalities=[Modality.VISION],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    requires_gpu=True,
    training_time="medium",
    memory_requirement="medium",
    sklearn_compatible=False,
    description="EfficientNet-B0 (Transfer Learning)",
    library="timm",
))

ModelRegistry.register(ModelSpec(
    name="mobilenet_v3",
    family=ModelFamily.CNN,
    modalities=[Modality.VISION],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    requires_gpu=True,
    training_time="fast",
    memory_requirement="low",
    sklearn_compatible=False,
    description="MobileNetV3 - Efficient for mobile/edge",
    library="timm",
))

ModelRegistry.register(ModelSpec(
    name="vit_base",
    family=ModelFamily.TRANSFORMER,
    modalities=[Modality.VISION],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    requires_gpu=True,
    training_time="slow",
    memory_requirement="high",
    sklearn_compatible=False,
    description="Vision Transformer (ViT) Base",
    library="timm",
))

# ============================================================================
# Register NLP Models
# ============================================================================

ModelRegistry.register(ModelSpec(
    name="tfidf_logistic",
    family=ModelFamily.CLASSICAL,
    modalities=[Modality.NLP],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    description="TF-IDF + Logistic Regression baseline",
    library="sklearn",
))

ModelRegistry.register(ModelSpec(
    name="tfidf_svm",
    family=ModelFamily.CLASSICAL,
    modalities=[Modality.NLP],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    description="TF-IDF + SVM baseline",
    library="sklearn",
))

ModelRegistry.register(ModelSpec(
    name="bert_base",
    family=ModelFamily.TRANSFORMER,
    modalities=[Modality.NLP],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS, TaskType.NER, TaskType.QA],
    requires_gpu=True,
    training_time="slow",
    memory_requirement="high",
    sklearn_compatible=False,
    description="BERT Base for NLP tasks",
    library="transformers",
))

ModelRegistry.register(ModelSpec(
    name="distilbert",
    family=ModelFamily.TRANSFORMER,
    modalities=[Modality.NLP],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    requires_gpu=True,
    training_time="medium",
    memory_requirement="medium",
    sklearn_compatible=False,
    description="DistilBERT - Faster, lighter BERT",
    library="transformers",
))

ModelRegistry.register(ModelSpec(
    name="roberta",
    family=ModelFamily.TRANSFORMER,
    modalities=[Modality.NLP],
    task_types=[TaskType.CLASSIFICATION, TaskType.MULTICLASS],
    requires_gpu=True,
    training_time="slow",
    memory_requirement="high",
    sklearn_compatible=False,
    description="RoBERTa - Robustly optimized BERT",
    library="transformers",
))


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create(model_name: str, task_type: str = "classification", **kwargs) -> Any:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model
            task_type: Task type (classification, regression)
            **kwargs: Override default parameters
            
        Returns:
            Model instance
        """
        spec = ModelRegistry.get(model_name)
        if spec is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        params = {**spec.default_params, **kwargs}
        
        # Sklearn models
        if model_name == "linear_regression":
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)
        
        elif model_name == "ridge_regression":
            from sklearn.linear_model import Ridge
            return Ridge(**params)
        
        elif model_name == "lasso_regression":
            from sklearn.linear_model import Lasso
            return Lasso(**params)
        
        elif model_name == "elastic_net":
            from sklearn.linear_model import ElasticNet
            return ElasticNet(**params)
        
        elif model_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
        
        elif model_name == "knn_classifier":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**params)
        
        elif model_name == "knn_regressor":
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(**params)
        
        elif model_name == "svc":
            from sklearn.svm import SVC
            return SVC(**params)
        
        elif model_name == "svr":
            from sklearn.svm import SVR
            return SVR(**params)
        
        elif model_name == "gaussian_nb":
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(**params)
        
        elif model_name == "multinomial_nb":
            from sklearn.naive_bayes import MultinomialNB
            return MultinomialNB(**params)
        
        elif model_name == "decision_tree_classifier":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(**params)
        
        elif model_name == "decision_tree_regressor":
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor(**params)
        
        elif model_name == "random_forest_classifier":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        
        elif model_name == "random_forest_regressor":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)
        
        elif model_name == "extra_trees_classifier":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(**params)
        
        elif model_name == "gradient_boosting_classifier":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**params)
        
        elif model_name == "hist_gradient_boosting_classifier":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(**params)
        
        elif model_name == "adaboost_classifier":
            from sklearn.ensemble import AdaBoostClassifier
            return AdaBoostClassifier(**params)
        
        elif model_name == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            return IsolationForest(**params)
        
        elif model_name == "one_class_svm":
            from sklearn.svm import OneClassSVM
            return OneClassSVM(**params)
        
        # XGBoost
        elif model_name == "xgboost":
            import xgboost as xgb
            if task_type == "regression":
                return xgb.XGBRegressor(**params)
            return xgb.XGBClassifier(**params)
        
        # LightGBM
        elif model_name == "lightgbm":
            import lightgbm as lgb
            if task_type == "regression":
                return lgb.LGBMRegressor(**params)
            return lgb.LGBMClassifier(**params)
        
        # CatBoost
        elif model_name == "catboost":
            import catboost as cb
            if task_type == "regression":
                return cb.CatBoostRegressor(**params)
            return cb.CatBoostClassifier(**params)
        
        else:
            raise ValueError(f"Model {model_name} not implemented in factory")
