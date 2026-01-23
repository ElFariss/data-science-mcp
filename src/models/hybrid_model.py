"""Hybrid Model - Combine multiple models for better predictions."""

from typing import Any, Optional
import numpy as np


class HybridModel:
    """
    Hybrid model combining a base model with a residual model.
    
    Useful for time series where you combine statistical (ARIMA) 
    with ML (XGBoost) approaches.
    """
    
    def __init__(
        self,
        base_model: Any,
        residual_model: Any,
        base_model_name: str = "base",
        residual_model_name: str = "residual",
    ):
        """
        Initialize hybrid model.
        
        Args:
            base_model: Primary model (e.g., ARIMA, Prophet)
            residual_model: Model to fit residuals (e.g., XGBoost)
            base_model_name: Name for the base model
            residual_model_name: Name for the residual model
        """
        self.base_model = base_model
        self.residual_model = residual_model
        self.base_model_name = base_model_name
        self.residual_model_name = residual_model_name
        self._fitted = False
    
    def fit(self, X: Any, y: Any) -> "HybridModel":
        """
        Fit the hybrid model.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            self
        """
        # Fit base model
        self.base_model.fit(X, y)
        
        # Get base predictions
        base_predictions = self.base_model.predict(X)
        
        # Calculate residuals
        residuals = y - base_predictions
        
        # Fit residual model on the residuals
        self.residual_model.fit(X, residuals)
        
        self._fitted = True
        return self
    
    def predict(self, X: Any) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Combined predictions
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get base predictions
        base_pred = self.base_model.predict(X)
        
        # Get residual predictions
        residual_pred = self.residual_model.predict(X)
        
        # Combine
        return base_pred + residual_pred
    
    def get_components(self, X: Any) -> dict:
        """
        Get individual component predictions.
        
        Args:
            X: Features
            
        Returns:
            Dict with base and residual predictions
        """
        return {
            "base": self.base_model.predict(X),
            "residual": self.residual_model.predict(X),
            "combined": self.predict(X),
        }


class StackingEnsemble:
    """
    Stacking ensemble combining multiple base models with a meta-learner.
    """
    
    def __init__(
        self,
        base_models: list[tuple[str, Any]],
        meta_model: Any,
        use_probabilities: bool = True,
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of (name, model) tuples
            meta_model: Model to combine base predictions
            use_probabilities: Use probabilities for classification
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_probabilities = use_probabilities
        self._fitted = False
    
    def fit(self, X: Any, y: Any) -> "StackingEnsemble":
        """
        Fit the stacking ensemble.
        
        Uses out-of-fold predictions to prevent overfitting.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            self
        """
        from sklearn.model_selection import KFold
        
        n_samples = len(X)
        n_models = len(self.base_models)
        
        # Create matrix for meta-features
        if self.use_probabilities and hasattr(self.base_models[0][1], 'predict_proba'):
            # For classification with probabilities
            meta_features = np.zeros((n_samples, n_models))
        else:
            meta_features = np.zeros((n_samples, n_models))
        
        # K-fold cross-validation for OOF predictions
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_fold_train = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
            y_fold_train = y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx]
            X_fold_val = X[val_idx] if isinstance(X, np.ndarray) else X.iloc[val_idx]
            
            for i, (name, model) in enumerate(self.base_models):
                model.fit(X_fold_train, y_fold_train)
                
                if self.use_probabilities and hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X_fold_val)
                    if preds.shape[1] == 2:
                        meta_features[val_idx, i] = preds[:, 1]
                    else:
                        meta_features[val_idx, i] = preds.max(axis=1)
                else:
                    meta_features[val_idx, i] = model.predict(X_fold_val)
        
        # Fit meta-model on OOF predictions
        self.meta_model.fit(meta_features, y)
        
        # Refit base models on full data
        for name, model in self.base_models:
            model.fit(X, y)
        
        self._fitted = True
        return self
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Get probability predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        meta_features = self._get_meta_features(X)
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        return self.meta_model.predict(meta_features)
    
    def _get_meta_features(self, X: Any) -> np.ndarray:
        """Get meta-features from base models."""
        n_samples = len(X)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        for i, (name, model) in enumerate(self.base_models):
            if self.use_probabilities and hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)
                if preds.shape[1] == 2:
                    meta_features[:, i] = preds[:, 1]
                else:
                    meta_features[:, i] = preds.max(axis=1)
            else:
                meta_features[:, i] = model.predict(X)
        
        return meta_features


class WeightedAverageEnsemble:
    """Simple weighted average ensemble."""
    
    def __init__(
        self,
        models: list[tuple[str, Any]],
        weights: Optional[list[float]] = None,
    ):
        """
        Initialize weighted ensemble.
        
        Args:
            models: List of (name, model) tuples
            weights: Optional weights (uniform if None)
        """
        self.models = models
        self.weights = weights
        self._fitted = False
    
    def fit(self, X: Any, y: Any) -> "WeightedAverageEnsemble":
        """Fit all models."""
        for name, model in self.models:
            model.fit(X, y)
        
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        self._fitted = True
        return self
    
    def predict(self, X: Any) -> np.ndarray:
        """Get weighted average predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        for (name, model), weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Get weighted average probabilities."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        probas = []
        for (name, model), weight in zip(self.models, self.weights):
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X) * weight
                probas.append(prob)
        
        if probas:
            return np.sum(probas, axis=0)
        return self.predict(X)
