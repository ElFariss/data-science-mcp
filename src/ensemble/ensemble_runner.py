"""Ensemble Creation Pipeline."""

from typing import Any, Optional
import numpy as np

from src.core.version_manager import VersionManager


async def run_ensemble_creation(
    versions: list[str],
    method: str = "average",
    weights: Optional[list[float]] = None,
    version_manager: Optional[VersionManager] = None,
    ctx: Any = None,
) -> dict:
    """
    Create ensemble from multiple model versions.
    
    Args:
        versions: List of version IDs to ensemble
        method: Ensemble method (average, weighted, stacking)
        weights: Optional weights for weighted averaging
        version_manager: Version manager instance
        ctx: MCP context for progress
        
    Returns:
        Ensemble results dictionary
    """
    if version_manager is None:
        version_manager = VersionManager()
    
    if len(versions) < 2:
        return {"error": "Need at least 2 versions to ensemble"}
    
    if ctx:
        await ctx.info(f"Creating {method} ensemble from {len(versions)} versions")
        await ctx.report_progress(progress=0.1, total=1.0, message="Loading models...")
    
    # Load predictions from each version
    all_predictions = []
    all_scores = []
    
    for version in versions:
        metadata = version_manager.get_metadata(version)
        if metadata is None:
            continue
        
        all_scores.append(metadata.best_score)
        
        # Load predictions
        version_path = version_manager.get_version_path(version)
        pred_path = version_path / "val_predictions.npy"
        
        if pred_path.exists():
            preds = np.load(pred_path)
            all_predictions.append(preds)
    
    if len(all_predictions) < 2:
        return {"error": "Not enough valid predictions found"}
    
    if ctx:
        await ctx.report_progress(progress=0.5, total=1.0, message="Combining predictions...")
    
    # Determine weights
    if method == "weighted" and weights is None:
        # Use scores as weights
        weights = np.array(all_scores)
        weights = weights / weights.sum()
    elif method == "average":
        weights = np.ones(len(all_predictions)) / len(all_predictions)
    
    # Combine predictions
    if method in ["average", "weighted"]:
        ensemble_pred = np.zeros_like(all_predictions[0])
        for pred, w in zip(all_predictions, weights):
            ensemble_pred += pred * w
    else:
        # Stacking would require meta-learner
        ensemble_pred = np.mean(all_predictions, axis=0)
    
    if ctx:
        await ctx.report_progress(progress=0.8, total=1.0, message="Saving ensemble...")
    
    # Create new version for ensemble
    ensemble_version = version_manager.create_new_version()
    
    # Save ensemble predictions
    ensemble_path = version_manager.get_version_path(ensemble_version)
    np.save(ensemble_path / "ensemble_predictions.npy", ensemble_pred)
    
    # Update metadata
    version_manager.update_metadata(
        ensemble_version,
        notes=f"Ensemble of {', '.join(versions)} using {method}",
    )
    
    if ctx:
        await ctx.info(f"Ensemble created as version {ensemble_version}")
        await ctx.report_progress(progress=1.0, total=1.0, message="Done")
    
    return {
        "ensemble_version": ensemble_version,
        "component_versions": versions,
        "method": method,
        "weights": weights.tolist() if isinstance(weights, np.ndarray) else weights,
    }
