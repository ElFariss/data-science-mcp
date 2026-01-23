"""Model Evaluation and Error Analysis."""

from typing import Any, Optional
from pathlib import Path
import numpy as np

from src.core.version_manager import VersionManager


async def run_model_evaluation(
    version: str,
    analysis_type: str = "full",
    version_manager: Optional[VersionManager] = None,
    ctx: Any = None,
) -> dict:
    """
    Run model evaluation and error analysis.
    
    Args:
        version: Version to evaluate
        analysis_type: "full", "quick", or "failures_only"
        version_manager: Version manager instance
        ctx: MCP context for progress
        
    Returns:
        Evaluation results dictionary
    """
    if version_manager is None:
        version_manager = VersionManager()
    
    if ctx:
        await ctx.info(f"Evaluating model version: {version}")
        await ctx.report_progress(progress=0.1, total=1.0, message="Loading model...")
    
    # Get version metadata
    metadata = version_manager.get_metadata(version)
    if metadata is None:
        return {"error": f"Version {version} not found"}
    
    # Load experiment data
    exp_data = version_manager.load_experiment_data(version)
    
    results = {
        "version": version,
        "model_name": metadata.best_model,
        "best_score": metadata.best_score,
        "metric_name": metadata.metric_name,
    }
    
    if ctx:
        await ctx.report_progress(progress=0.5, total=1.0, message="Computing metrics...")
    
    # Load predictions if available
    version_path = version_manager.get_version_path(version)
    predictions_path = version_path / "predictions.npy"
    labels_path = version_path / "labels.npy"
    
    if predictions_path.exists() and labels_path.exists():
        y_pred = np.load(predictions_path)
        y_true = np.load(labels_path)
        
        # Compute detailed metrics
        results["metrics"] = compute_metrics(y_true, y_pred, metadata.task_type)
        
        if analysis_type in ["full", "failures_only"]:
            results["error_analysis"] = analyze_errors(y_true, y_pred)
    
    if ctx:
        await ctx.report_progress(progress=0.9, total=1.0, message="Generating recommendations...")
    
    # Generate recommendations
    results["recommendations"] = generate_recommendations(results, metadata)
    
    if ctx:
        await ctx.info("Evaluation complete")
        await ctx.report_progress(progress=1.0, total=1.0, message="Done")
    
    return results


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict:
    """Compute comprehensive metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
    )
    
    if "regression" in task_type.lower():
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    else:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
        
        try:
            metrics["precision"] = float(precision_score(y_true, y_pred, average='weighted'))
            metrics["recall"] = float(recall_score(y_true, y_pred, average='weighted'))
            metrics["f1"] = float(f1_score(y_true, y_pred, average='weighted'))
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        except Exception:
            pass
        
        return metrics


def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Analyze error patterns."""
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]
    
    return {
        "total_errors": int(errors.sum()),
        "error_rate": float(errors.mean()),
        "error_indices": error_indices[:100].tolist(),  # First 100
    }


def generate_recommendations(results: dict, metadata: Any) -> list[str]:
    """Generate recommendations based on evaluation."""
    recommendations = []
    
    metrics = results.get("metrics", {})
    
    if "accuracy" in metrics:
        if metrics["accuracy"] < 0.7:
            recommendations.append("Consider trying more complex models or feature engineering")
        if metrics.get("precision", 0) < metrics.get("recall", 0):
            recommendations.append("Model has high recall but low precision - may be over-predicting positive class")
    
    if "r2" in metrics:
        if metrics["r2"] < 0.5:
            recommendations.append("Low R² score - consider adding more features or trying non-linear models")
    
    recommendations.append("Consider hyperparameter tuning with Optuna")
    recommendations.append("Try ensemble of top models for potential improvement")
    
    return recommendations
