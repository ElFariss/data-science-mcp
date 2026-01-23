"""Submission File Generator."""

from typing import Any, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.core.version_manager import VersionManager


async def generate_submission_file(
    version: str,
    test_data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    version_manager: Optional[VersionManager] = None,
    ctx: Any = None,
) -> str:
    """
    Generate validated submission file.
    
    Args:
        version: Version to use for predictions
        test_data_path: Path to test data
        output_path: Output file path
        version_manager: Version manager instance
        ctx: MCP context for progress
        
    Returns:
        Path to generated submission file
    """
    if version_manager is None:
        version_manager = VersionManager()
    
    if ctx:
        await ctx.info(f"Generating submission for version {version}")
        await ctx.report_progress(progress=0.1, total=1.0, message="Loading model...")
    
    # Get version info
    metadata = version_manager.get_metadata(version)
    if metadata is None:
        raise ValueError(f"Version {version} not found")
    
    # Load model
    try:
        model = version_manager.load_model(version, metadata.best_model)
    except FileNotFoundError:
        raise ValueError(f"Model not found for version {version}")
    
    if ctx:
        await ctx.report_progress(progress=0.3, total=1.0, message="Loading test data...")
    
    # Load test data
    if test_data_path is None:
        # Try to find test.csv in same directory as training data
        if metadata.data_path:
            train_path = Path(metadata.data_path)
            test_path = train_path.parent / "test.csv"
            if test_path.exists():
                test_data_path = str(test_path)
    
    if test_data_path is None:
        raise ValueError("No test data path provided and could not auto-detect")
    
    path = Path(test_data_path)
    if path.suffix == '.parquet':
        test_df = pd.read_parquet(path)
    else:
        test_df = pd.read_csv(path)
    
    if ctx:
        await ctx.report_progress(progress=0.5, total=1.0, message="Making predictions...")
    
    # Get ID column if exists
    id_col = None
    for col in ['id', 'Id', 'ID', 'index']:
        if col in test_df.columns:
            id_col = col
            break
    
    # Prepare features
    X_test = test_df.copy()
    if id_col:
        ids = X_test[id_col]
        X_test = X_test.drop(columns=[id_col])
    else:
        ids = pd.Series(range(len(test_df)))
    
    # Handle categorical features (same as training)
    for col in X_test.select_dtypes(include=['object']).columns:
        X_test[col] = X_test[col].astype('category').cat.codes
    
    # Make predictions
    predictions = model.predict(X_test)
    
    if ctx:
        await ctx.report_progress(progress=0.8, total=1.0, message="Saving submission...")
    
    # Create submission dataframe
    submission = pd.DataFrame({
        id_col if id_col else 'id': ids,
        metadata.target_column or 'target': predictions
    })
    
    # Validate
    validation_errors = validate_submission(submission, metadata)
    if validation_errors:
        for error in validation_errors:
            if ctx:
                await ctx.warning(f"Validation: {error}")
    
    # Save submission
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"submissions/submission_{version}_{timestamp}.csv"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    # Also save to version folder
    version_path = version_manager.get_version_path(version)
    submission.to_csv(version_path / "submission.csv", index=False)
    
    if ctx:
        await ctx.info(f"Submission saved to {output_path}")
        await ctx.report_progress(progress=1.0, total=1.0, message="Done")
    
    return output_path


def validate_submission(submission: pd.DataFrame, metadata: Any) -> list[str]:
    """Validate submission file."""
    errors = []
    
    # Check for missing predictions
    if submission.isnull().any().any():
        errors.append("Submission contains missing values")
    
    # Check row count
    if len(submission) == 0:
        errors.append("Submission is empty")
    
    # Check columns
    if len(submission.columns) < 2:
        errors.append("Submission should have at least 2 columns (id + prediction)")
    
    return errors
