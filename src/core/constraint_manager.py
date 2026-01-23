"""Constraint Manager for time budget and resource limits.

Handles time budget enforcement, compute resource monitoring,
and model preference constraints (e.g., basic ML only).
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class ResourceLimits:
    """Resource limits for training."""
    max_memory_gb: Optional[float] = None
    max_gpu_memory_gb: Optional[float] = None
    max_parallel_jobs: int = -1  # -1 = auto


class ConstraintManager:
    """Manages time budget and resource constraints for training."""
    
    # Basic ML models that are allowed when basic_ml_only is True
    BASIC_ML_MODELS = frozenset([
        # Linear models
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "elastic_net",
        "logistic_regression",
        "bayesian_linear_regression",
        
        # Distance-based
        "knn",
        "knn_classifier",
        "knn_regressor",
        "radius_neighbors",
        
        # SVM
        "svm",
        "linear_svm",
        "kernel_svm",
        "svr",
        
        # Naive Bayes
        "naive_bayes",
        "gaussian_nb",
        "multinomial_nb",
        "bernoulli_nb",
        
        # Trees
        "decision_tree",
        "decision_tree_classifier",
        "decision_tree_regressor",
        
        # Ensembles (non-boosting)
        "random_forest",
        "random_forest_classifier",
        "random_forest_regressor",
        "extra_trees",
        "extra_trees_classifier",
        "extra_trees_regressor",
        "bagging",
        
        # Boosting (considered basic for tabular)
        "gradient_boosting",
        "adaboost",
        "xgboost",
        "lightgbm",
        "catboost",
        "histogram_gradient_boosting",
    ])
    
    # Models that require deep learning / GPU
    DEEP_LEARNING_MODELS = frozenset([
        # Tabular neural networks
        "mlp",
        "tabnet",
        "tab_transformer",
        "ft_transformer",
        "saint",
        "node",
        "deep_gbm",
        "danets",
        "autoint",
        "deepfm",
        "xdeepfm",
        "dcn",
        
        # Time series deep learning
        "lstm",
        "gru",
        "rnn",
        "tcn",
        "temporal_fusion_transformer",
        "informer",
        "autoformer",
        "fedformer",
        "patchtst",
        "timesnet",
        "deepar",
        "deepstate",
        "deepvar",
        "neuralprophet",
        
        # Vision models
        "lenet",
        "alexnet",
        "vgg",
        "resnet",
        "resnext",
        "densenet",
        "mobilenet",
        "shufflenet",
        "squeezenet",
        "efficientnet",
        "convnext",
        "regnet",
        "vit",
        "deit",
        "swin",
        "cvt",
        "maxvit",
        "coatnet",
        "yolo",
        "faster_rcnn",
        "mask_rcnn",
        "detr",
        "unet",
        "deeplab",
        "sam",
        
        # NLP models
        "bert",
        "roberta",
        "albert",
        "distilbert",
        "deberta",
        "electra",
        "gpt2",
        "gpt_neo",
        "gpt_j",
        "t5",
        "bart",
        "pegasus",
        "bilstm_crf",
    ])
    
    def __init__(self):
        """Initialize constraint manager."""
        self.start_time: Optional[float] = None
        self.time_budget_seconds: Optional[float] = None
        self.deep_learning_allowed: bool = True
        self.basic_ml_only: bool = False
        self.resource_limits = ResourceLimits()
        self._on_time_warning_callbacks: list[Callable] = []
        self._warned_at_percentages: set[int] = set()
    
    def set_time_budget(self, hours: float) -> None:
        """
        Set the time budget for training.
        
        Args:
            hours: Time budget in hours
        """
        self.time_budget_seconds = hours * 3600
        self.start_time = time.time()
        self._warned_at_percentages = set()
    
    def set_deep_learning_allowed(self, allowed: bool) -> None:
        """Set whether deep learning models are allowed."""
        self.deep_learning_allowed = allowed
    
    def set_basic_ml_only(self, basic_only: bool) -> None:
        """Set whether only basic ML models should be used."""
        self.basic_ml_only = basic_only
    
    def get_remaining_time(self) -> float:
        """
        Get remaining time in seconds.
        
        Returns:
            Remaining time, or infinity if no budget set
        """
        if self.time_budget_seconds is None or self.start_time is None:
            return float('inf')
        
        elapsed = time.time() - self.start_time
        return max(0, self.time_budget_seconds - elapsed)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_progress_percentage(self) -> float:
        """
        Get time progress as percentage (0-100).
        
        Returns:
            Percentage of time budget used
        """
        if self.time_budget_seconds is None or self.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return min(100, (elapsed / self.time_budget_seconds) * 100)
    
    def check_time_exceeded(self) -> bool:
        """
        Check if time budget has been exceeded.
        
        Returns:
            True if time is up, False otherwise
        """
        return self.get_remaining_time() <= 0
    
    def check_time_warning(self, threshold_percent: int = 75) -> bool:
        """
        Check if we should warn about time running low.
        
        Args:
            threshold_percent: Percentage threshold for warning
            
        Returns:
            True if warning should be issued (only once per threshold)
        """
        progress = self.get_progress_percentage()
        
        if progress >= threshold_percent and threshold_percent not in self._warned_at_percentages:
            self._warned_at_percentages.add(threshold_percent)
            return True
        
        return False
    
    def is_model_allowed(self, model_name: str) -> bool:
        """
        Check if a model is allowed under current constraints.
        
        Args:
            model_name: Name of the model (lowercase)
            
        Returns:
            True if model is allowed, False otherwise
        """
        model_name = model_name.lower().replace("-", "_").replace(" ", "_")
        
        # Check deep learning constraint
        if not self.deep_learning_allowed:
            if model_name in self.DEEP_LEARNING_MODELS:
                return False
        
        # Check basic ML only constraint
        if self.basic_ml_only:
            if model_name not in self.BASIC_ML_MODELS:
                return False
        
        return True
    
    def filter_allowed_models(self, model_names: list[str]) -> list[str]:
        """
        Filter a list of models to only those allowed.
        
        Args:
            model_names: List of model names
            
        Returns:
            Filtered list of allowed models
        """
        return [m for m in model_names if self.is_model_allowed(m)]
    
    def estimate_can_train(
        self,
        model_name: str,
        estimated_minutes: float
    ) -> bool:
        """
        Check if there's enough time to train a model.
        
        Args:
            model_name: Name of the model
            estimated_minutes: Estimated training time in minutes
            
        Returns:
            True if there's likely enough time
        """
        if not self.is_model_allowed(model_name):
            return False
        
        remaining_seconds = self.get_remaining_time()
        estimated_seconds = estimated_minutes * 60
        
        # Add 20% buffer
        return remaining_seconds >= estimated_seconds * 1.2
    
    def get_constraints_summary(self) -> dict:
        """Get a summary of current constraints."""
        return {
            "time_budget_hours": (
                self.time_budget_seconds / 3600 
                if self.time_budget_seconds else None
            ),
            "remaining_time_hours": round(self.get_remaining_time() / 3600, 2),
            "elapsed_time_hours": round(self.get_elapsed_time() / 3600, 2),
            "progress_percent": round(self.get_progress_percentage(), 1),
            "deep_learning_allowed": self.deep_learning_allowed,
            "basic_ml_only": self.basic_ml_only,
            "time_exceeded": self.check_time_exceeded(),
        }
    
    def reset(self) -> None:
        """Reset all constraints to defaults."""
        self.start_time = None
        self.time_budget_seconds = None
        self.deep_learning_allowed = True
        self.basic_ml_only = False
        self._warned_at_percentages = set()
