"""Training Pipeline - High-level training workflows."""

from typing import Any, Optional
from pathlib import Path
import pandas as pd

from src.core.version_manager import VersionManager
from src.core.constraint_manager import ConstraintManager
from src.core.dataset_detector import DatasetDetector
from src.models.model_selector import ModelSelector
from src.models.multi_model import MultiModelRunner


async def run_automatic_training(
    data_path: str,
    target_column: Optional[str] = None,
    version_manager: Optional[VersionManager] = None,
    constraint_manager: Optional[ConstraintManager] = None,
    ctx: Any = None,
) -> dict:
    """
    Run fully automatic training workflow.
    
    Args:
        data_path: Path to training data
        target_column: Target column name (auto-detect if None)
        version_manager: Version manager instance
        constraint_manager: Constraint manager instance
        ctx: MCP context for progress reporting
        
    Returns:
        Training results dictionary
    """
    from sklearn.model_selection import train_test_split
    
    if version_manager is None:
        version_manager = VersionManager()
    
    if constraint_manager is None:
        constraint_manager = ConstraintManager()
    
    # Create new version
    version = version_manager.create_new_version()
    
    if ctx:
        await ctx.info(f"Starting automatic training for version {version}")
        await ctx.report_progress(progress=0.05, total=1.0, message="Loading data...")
    
    # Load data
    path = Path(data_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    # Detect dataset characteristics
    detector = DatasetDetector()
    dataset_info = detector.analyze_dataset(data_path, target_column)
    modality = dataset_info.get("modality", "tabular")
    task_type = dataset_info.get("task_type", "classification")
    
    # Find target column
    if target_column is None:
        target_column = dataset_info.get("target_column", df.columns[-1])
    
    if ctx:
        await ctx.report_progress(progress=0.1, total=1.0, message="Preparing data...")
    
    # Prepare features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical features (simple encoding for now)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if ctx:
        await ctx.report_progress(progress=0.15, total=1.0, message="Selecting models...")
    
    # Select models
    selector = ModelSelector(constraint_manager)
    if constraint_manager.basic_ml_only:
        selector.restrict_to_basic_ml(True)
    
    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass
    
    models = selector.select_models(
        modality=modality,
        task_type=task_type,
        n_samples=len(df),
        n_features=len(X.columns),
        has_gpu=has_gpu,
        max_models=5,
    )
    
    if ctx:
        await ctx.info(f"Selected models: {', '.join(models)}")
        await ctx.report_progress(progress=0.2, total=1.0, message="Training models...")
    
    # Train models
    runner = MultiModelRunner(constraint_manager)
    
    is_regression = "regression" in task_type.lower()
    metric_name = "r2" if is_regression else "accuracy"
    
    async def progress_cb(msg: str, pct: float):
        if ctx:
            await ctx.report_progress(
                progress=0.2 + pct * 0.6,
                total=1.0,
                message=msg
            )
    
    results = await runner.train_multiple(
        models=models,
        X_train=X_train.values,
        y_train=y_train.values,
        X_val=X_val.values,
        y_val=y_val.values,
        task_type="regression" if is_regression else "classification",
        metric_name=metric_name,
        progress_callback=progress_cb,
    )
    
    if ctx:
        await ctx.report_progress(progress=0.85, total=1.0, message="Saving results...")
    
    # Get best model
    best = runner.get_best_model()
    rankings = runner.get_rankings()
    
    # Save results
    if best:
        model_path = version_manager.save_model(version, best.model, best.model_name)
        
        version_manager.update_metadata(
            version,
            modality=modality,
            task_type=task_type,
            models_trained=[r.model_name for r in results if r.error is None],
            best_model=best.model_name,
            best_score=best.score,
            metric_name=metric_name,
            data_path=str(data_path),
            target_column=target_column,
        )
        
        # Save model plan
        model_plan = runner.generate_model_plan()
        plan_path = version_manager.get_version_path(version) / "model_plan.md"
        with open(plan_path, 'w') as f:
            f.write(model_plan)
    
    if ctx:
        await ctx.info(f"Training complete. Best model: {best.model_name if best else 'None'}")
        await ctx.report_progress(progress=1.0, total=1.0, message="Done")
    
    # Prepare return value
    return {
        "version": version,
        "models_trained": [r.model_name for r in results if r.error is None],
        "best_model": best.model_name if best else None,
        "best_score": round(best.score, 4) if best else None,
        "metric_name": metric_name,
        "rankings": rankings,
        "training_time_seconds": sum(r.training_time for r in results),
        "next_recommendations": [
            f"Try hyperparameter tuning on {best.model_name}" if best else "",
            "Consider ensembling top models",
            "Run /evaluate for error analysis",
        ],
    }


async def run_eda_guided_training(
    data_path: str,
    eda_report_path: str,
    target_column: Optional[str] = None,
    version_manager: Optional[VersionManager] = None,
    constraint_manager: Optional[ConstraintManager] = None,
    ctx: Any = None,
) -> dict:
    """
    Train models using EDA report recommendations.
    """
    # Parse EDA report for recommendations
    # For now, fall back to automatic training
    return await run_automatic_training(
        data_path=data_path,
        target_column=target_column,
        version_manager=version_manager,
        constraint_manager=constraint_manager,
        ctx=ctx,
    )


async def run_benchmark_training(
    data_path: str,
    target_column: Optional[str] = None,
    models_to_test: Optional[list[str]] = None,
    version_manager: Optional[VersionManager] = None,
    ctx: Any = None,
) -> str:
    """
    Quick benchmark training on subset of data.
    
    Returns path to model_plan.md
    """
    from sklearn.model_selection import train_test_split
    
    if version_manager is None:
        version_manager = VersionManager()
    
    if ctx:
        await ctx.info("Running quick benchmark training...")
    
    # Load data
    path = Path(data_path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    # Use subset
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)
    
    # Detect task type
    detector = DatasetDetector()
    dataset_info = detector.analyze_dataset(data_path, target_column)
    task_type = dataset_info.get("task_type", "classification")
    
    if target_column is None:
        target_column = dataset_info.get("target_column", df.columns[-1])
    
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Select models to test
    if models_to_test is None:
        models_to_test = [
            "logistic_regression" if "classification" in task_type else "linear_regression",
            "random_forest_classifier" if "classification" in task_type else "random_forest_regressor",
            "lightgbm",
            "xgboost",
        ]
    
    # Quick training without time constraints
    constraint = ConstraintManager()
    constraint.set_time_budget(0.5)  # 30 minutes max for benchmark
    
    runner = MultiModelRunner(constraint)
    
    is_regression = "regression" in task_type.lower()
    
    await runner.train_multiple(
        models=models_to_test,
        X_train=X_train.values,
        y_train=y_train.values,
        X_val=X_val.values,
        y_val=y_val.values,
        task_type="regression" if is_regression else "classification",
    )
    
    # Generate model plan
    model_plan = runner.generate_model_plan()
    
    # Save to reports
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_path = reports_dir / f"model_plan_{timestamp}.md"
    
    with open(plan_path, 'w') as f:
        f.write(model_plan)
    
    if ctx:
        await ctx.info(f"Benchmark complete. Model plan saved to {plan_path}")
    
    return str(plan_path)
