"""Feature Engineering Pipeline."""

from typing import Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.core.version_manager import VersionManager
from src.features.tabular_features import TabularFeatureEngineer


async def run_feature_engineering(
    data_path: str,
    strategy: str = "basic",
    version_manager: Optional[VersionManager] = None,
    ctx: Any = None,
) -> dict:
    """
    Run feature engineering pipeline.
    
    Args:
        data_path: Path to raw data
        strategy: Feature engineering strategy
        version_manager: Version manager instance
        ctx: MCP context for progress
        
    Returns:
        Feature engineering results
    """
    if version_manager is None:
        version_manager = VersionManager()
    
    if ctx:
        await ctx.info(f"Running feature engineering with strategy: {strategy}")
        await ctx.report_progress(progress=0.1, total=1.0, message="Loading data...")
    
    # Load data
    path = Path(data_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    # Assume last column is target
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if ctx:
        await ctx.report_progress(progress=0.3, total=1.0, message="Transforming features...")
    
    # Create feature engineer
    fe = TabularFeatureEngineer(strategy=strategy)
    
    # Transform
    X_transformed = fe.fit_transform(X, y)
    
    if ctx:
        await ctx.report_progress(progress=0.7, total=1.0, message="Saving results...")
    
    # Create output directory
    features_dir = Path("features")
    features_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save transformed data
    output_path = features_dir / f"transformed_{strategy}_{timestamp}.parquet"
    
    # Combine with target
    output_df = X_transformed.copy()
    output_df[target_col] = y.values
    output_df.to_parquet(output_path, index=False)
    
    # Save config
    config_path = features_dir / f"config_{strategy}_{timestamp}.json"
    fe.save_config(str(config_path))
    
    # Update version if available
    latest_version = version_manager.get_latest_version()
    if latest_version:
        version_manager.update_metadata(
            latest_version,
            feature_config=str(config_path),
        )
        
        # Copy to version folder
        version_dir = version_manager.get_version_path(latest_version) / "features"
        version_dir.mkdir(exist_ok=True)
        
        import shutil
        shutil.copy(output_path, version_dir / "transformed_data.parquet")
        shutil.copy(config_path, version_dir / "feature_config.json")
    
    if ctx:
        await ctx.info(f"Feature engineering complete. Output: {output_path}")
        await ctx.report_progress(progress=1.0, total=1.0, message="Done")
    
    return {
        "version": latest_version,
        "strategy": strategy,
        "features_created": fe.get_feature_names(),
        "n_features_original": len(X.columns),
        "n_features_transformed": len(X_transformed.columns),
        "transformed_data_path": str(output_path),
        "feature_config_path": str(config_path),
    }
