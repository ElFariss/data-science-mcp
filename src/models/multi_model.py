"""Multi-Model Runner - Train multiple models with time budget management."""

from typing import Any, Optional, Callable, Awaitable
from dataclasses import dataclass
import time
import asyncio

from src.core.constraint_manager import ConstraintManager
from src.models.registry import ModelFactory, ModelRegistry


@dataclass
class ModelResult:
    """Result from training a single model."""
    model_name: str
    score: float
    metric_name: str
    training_time: float
    model: Any = None
    predictions: Any = None
    params: dict = None
    error: Optional[str] = None


class MultiModelRunner:
    """Trains multiple models within time budget constraints."""
    
    def __init__(self, constraint_manager: ConstraintManager):
        """
        Initialize multi-model runner.
        
        Args:
            constraint_manager: Constraint manager for time budget
        """
        self.constraint_manager = constraint_manager
        self.results: list[ModelResult] = []
    
    async def train_multiple(
        self,
        models: list[str],
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        task_type: str = "classification",
        metric_name: str = "accuracy",
        progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None,
    ) -> list[ModelResult]:
        """
        Train multiple models within time budget.
        
        Args:
            models: List of model names to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            task_type: classification or regression
            metric_name: Metric to evaluate
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of ModelResult objects
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score
        )
        
        results = []
        
        for i, model_name in enumerate(models):
            # Check time budget
            if self.constraint_manager.check_time_exceeded():
                if progress_callback:
                    await progress_callback(
                        f"Time budget exceeded, stopping after {len(results)} models",
                        1.0
                    )
                break
            
            # Check if model is allowed
            if not self.constraint_manager.is_model_allowed(model_name):
                continue
            
            if progress_callback:
                await progress_callback(
                    f"Training {model_name} ({i+1}/{len(models)})",
                    (i + 0.5) / len(models)
                )
            
            try:
                # Create model
                model = ModelFactory.create(model_name, task_type=task_type)
                
                # Train
                start_time = time.time()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, model.fit, X_train, y_train)
                training_time = time.time() - start_time
                
                # Predict
                y_pred = await loop.run_in_executor(None, model.predict, X_val)
                
                # Evaluate
                if task_type == "regression":
                    if metric_name == "mse":
                        score = -mean_squared_error(y_val, y_pred)  # Negative for consistency
                    elif metric_name == "mae":
                        score = -mean_absolute_error(y_val, y_pred)
                    else:
                        score = r2_score(y_val, y_pred)
                        metric_name = "r2"
                else:
                    if metric_name == "f1":
                        score = f1_score(y_val, y_pred, average="weighted")
                    elif metric_name == "auc":
                        try:
                            # Also offload predict_proba if needed
                            y_proba = await loop.run_in_executor(None, lambda: model.predict_proba(X_val)[:, 1])
                            score = roc_auc_score(y_val, y_proba)
                        except:
                            score = accuracy_score(y_val, y_pred)
                            metric_name = "accuracy"
                    else:
                        score = accuracy_score(y_val, y_pred)
                        metric_name = "accuracy"
                
                result = ModelResult(
                    model_name=model_name,
                    score=score,
                    metric_name=metric_name,
                    training_time=training_time,
                    model=model,
                    predictions=y_pred,
                    params=model.get_params() if hasattr(model, 'get_params') else {},
                )
                results.append(result)
                
            except Exception as e:
                results.append(ModelResult(
                    model_name=model_name,
                    score=0.0,
                    metric_name=metric_name,
                    training_time=0.0,
                    error=str(e),
                ))
        
        self.results = results
        return results
    
    def get_best_model(self) -> Optional[ModelResult]:
        """Get the best performing model."""
        valid_results = [r for r in self.results if r.error is None and r.score > 0]
        if not valid_results:
            return None
        return max(valid_results, key=lambda r: r.score)
    
    def get_rankings(self) -> list[dict]:
        """Get ranked list of all models."""
        valid_results = [r for r in self.results if r.error is None]
        sorted_results = sorted(valid_results, key=lambda r: r.score, reverse=True)
        
        rankings = []
        for i, r in enumerate(sorted_results, 1):
            rankings.append({
                "rank": i,
                "model": r.model_name,
                "score": round(r.score, 4),
                "metric": r.metric_name,
                "training_time": round(r.training_time, 2),
            })
        return rankings
    
    def generate_model_plan(self) -> str:
        """
        Generate a model plan markdown document.
        
        Returns:
            Markdown string with model rankings and recommendations
        """
        rankings = self.get_rankings()
        best = self.get_best_model()
        
        lines = [
            "# Model Benchmarking Results",
            "",
            f"## Summary",
            f"- Models tested: {len(rankings)}",
            f"- Best model: {best.model_name if best else 'N/A'}",
            f"- Best score: {round(best.score, 4) if best else 'N/A'}",
            "",
            "## Rankings",
            "",
            "| Rank | Model | Score | Training Time |",
            "|------|-------|-------|---------------|",
        ]
        
        for r in rankings:
            lines.append(
                f"| {r['rank']} | {r['model']} | {r['score']} | {r['training_time']}s |"
            )
        
        lines.extend([
            "",
            "## Recommendations",
            "",
        ])
        
        if rankings:
            lines.append(f"1. Use **{rankings[0]['model']}** as primary model")
            if len(rankings) > 1:
                lines.append(f"2. Try **{rankings[1]['model']}** as alternative")
            if len(rankings) > 2:
                lines.append(f"3. Consider ensemble of top 2-3 models")
        
        return "\n".join(lines)
