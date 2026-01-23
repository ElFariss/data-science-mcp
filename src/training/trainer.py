"""Model Trainer with progress reporting and early stopping."""

from typing import Any, Optional, Callable, Awaitable
import time
import numpy as np

from src.core.constraint_manager import ConstraintManager


class ModelTrainer:
    """Generic model trainer with time budget awareness."""
    
    def __init__(
        self,
        model: Any,
        constraint_manager: Optional[ConstraintManager] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            constraint_manager: Optional constraint manager for time budget
        """
        self.model = model
        self.constraint_manager = constraint_manager
        self.training_history = []
        self.best_score = None
        self.best_model = None
    
    async def train(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        metric_fn: Optional[Callable] = None,
        progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None,
    ) -> dict:
        """
        Train the model with progress reporting.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Optional validation data
            metric_fn: Optional metric function(y_true, y_pred)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Training results dictionary
        """
        from sklearn.metrics import accuracy_score, mean_squared_error
        
        start_time = time.time()
        
        if progress_callback:
            await progress_callback(0.1, "Starting training...")
        
        # Check time budget
        if self.constraint_manager and self.constraint_manager.check_time_exceeded():
            return {
                "status": "skipped",
                "reason": "time_budget_exceeded",
            }
        
        try:
            # Fit model
            self.model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            if progress_callback:
                await progress_callback(0.7, "Evaluating...")
            
            # Evaluate on training data
            train_pred = self.model.predict(X_train)
            
            # Evaluate on validation data if provided
            val_score = None
            if X_val is not None and y_val is not None:
                val_pred = self.model.predict(X_val)
                
                if metric_fn:
                    val_score = metric_fn(y_val, val_pred)
                else:
                    # Auto-detect metric
                    if y_train.dtype in ['float64', 'float32']:
                        val_score = -mean_squared_error(y_val, val_pred)  # Negative MSE
                    else:
                        val_score = accuracy_score(y_val, val_pred)
            
            if progress_callback:
                await progress_callback(1.0, "Training complete")
            
            result = {
                "status": "success",
                "training_time": round(training_time, 2),
                "val_score": val_score,
            }
            
            self.training_history.append(result)
            
            if val_score is not None:
                if self.best_score is None or val_score > self.best_score:
                    self.best_score = val_score
                    self.best_model = self.model
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    def predict(self, X: Any) -> Any:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_model(self) -> Any:
        """Get the trained model."""
        return self.best_model if self.best_model else self.model


class CrossValidator:
    """Cross-validation with time budget awareness."""
    
    def __init__(
        self,
        model_factory: Callable,
        cv_strategy: str = "kfold",
        n_splits: int = 5,
        constraint_manager: Optional[ConstraintManager] = None,
    ):
        """
        Initialize cross-validator.
        
        Args:
            model_factory: Callable that returns a new model instance
            cv_strategy: CV strategy (kfold, stratified, timeseries)
            n_splits: Number of CV splits
            constraint_manager: Optional constraint manager
        """
        self.model_factory = model_factory
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.constraint_manager = constraint_manager
        self.fold_scores = []
    
    def get_cv_splitter(self, X: Any, y: Any):
        """Get appropriate CV splitter."""
        from sklearn.model_selection import (
            KFold, StratifiedKFold, TimeSeriesSplit
        )
        
        if self.cv_strategy == "stratified":
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        elif self.cv_strategy == "timeseries":
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
    
    async def cross_validate(
        self,
        X: Any,
        y: Any,
        metric_fn: Optional[Callable] = None,
        progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None,
    ) -> dict:
        """
        Perform cross-validation.
        
        Args:
            X, y: Data
            metric_fn: Metric function
            progress_callback: Progress callback
            
        Returns:
            CV results with mean and std of scores
        """
        from sklearn.metrics import accuracy_score, mean_squared_error
        
        cv = self.get_cv_splitter(X, y)
        self.fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Check time budget
            if self.constraint_manager and self.constraint_manager.check_time_exceeded():
                break
            
            if progress_callback:
                await progress_callback(
                    (fold_idx + 0.5) / self.n_splits,
                    f"Training fold {fold_idx + 1}/{self.n_splits}"
                )
            
            # Split data
            if hasattr(X, 'iloc'):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            
            if hasattr(y, 'iloc'):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = self.model_factory()
            model.fit(X_train, y_train)
            
            # Evaluate
            val_pred = model.predict(X_val)
            
            if metric_fn:
                score = metric_fn(y_val, val_pred)
            else:
                if hasattr(y_train, 'dtype') and y_train.dtype in ['float64', 'float32']:
                    score = -mean_squared_error(y_val, val_pred)
                else:
                    score = accuracy_score(y_val, val_pred)
            
            self.fold_scores.append(score)
        
        if progress_callback:
            await progress_callback(1.0, "CV complete")
        
        return {
            "cv_scores": self.fold_scores,
            "mean_score": np.mean(self.fold_scores) if self.fold_scores else 0,
            "std_score": np.std(self.fold_scores) if self.fold_scores else 0,
            "n_folds_completed": len(self.fold_scores),
        }
