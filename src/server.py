"""Main MCP Server for AI Data Science Agent.

This server provides tools for end-to-end machine learning workflows including:
- CSV dataset reading and analysis
- Automated EDA (Exploratory Data Analysis)
- Model training with 100+ architectures
- Feature engineering with comparison
- Time-budget constrained training loops
- Multi-model and hybrid model support
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional
import time

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field

from src.core.version_manager import VersionManager
from src.core.constraint_manager import ConstraintManager
from src.core.dataset_detector import DatasetDetector
from src.core.experiment_tracker import ExperimentTracker


# ============================================================================
# Application Context (Lifespan Resources)
# ============================================================================

@dataclass
class AppContext:
    """Application context with shared resources across tool calls."""
    version_manager: VersionManager = field(default_factory=VersionManager)
    constraint_manager: ConstraintManager = field(default_factory=ConstraintManager)
    dataset_detector: DatasetDetector = field(default_factory=DatasetDetector)
    experiment_tracker: ExperimentTracker = field(default_factory=ExperimentTracker)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with shared resources."""
    context = AppContext()
    try:
        yield context
    finally:
        # Cleanup on shutdown
        pass


# ============================================================================
# MCP Server Initialization
# ============================================================================

mcp = FastMCP(
    "AI Data Science Agent",
    instructions="""An autonomous AI agent for end-to-end machine learning workflows.
    
Available Commands:
- /system - Inspect hardware capabilities
- /plan - Analyze dataset and create strategy
- /eda - Run exploratory data analysis
- /train - Train models (automatic, EDA-guided, or plan-guided)
- /features - Create feature engineering pipelines
- /evaluate - Deep error analysis
- /ensemble - Combine multiple models
- /submit - Generate submission files

The agent supports:
- Tabular data (60+ models including XGBoost, LightGBM, CatBoost)
- Time series (30+ models including ARIMA, Prophet, LSTM)
- Computer vision (40+ models including ResNet, EfficientNet, ViT)
- NLP (50+ models including BERT, GPT, T5)
""",
    lifespan=app_lifespan,
)


# ============================================================================
# Elicitation Schemas
# ============================================================================

class TimeBudgetConfig(BaseModel):
    """Configuration for training time budget and constraints."""
    time_budget_hours: float = Field(
        default=3.0,
        description="Total time budget in hours for training loop",
        ge=0.1,
        le=168.0  # Max 1 week
    )
    allow_deep_learning: bool = Field(
        default=True,
        description="Allow GPU-based deep learning models"
    )
    basic_ml_only: bool = Field(
        default=False,
        description="Restrict to basic ML models only (Linear, RF, XGBoost, etc.)"
    )


class FeatureStrategyConfig(BaseModel):
    """Configuration for feature engineering strategy."""
    strategy: str = Field(
        default="basic",
        description="Feature engineering strategy: basic, advanced, target_encoding, or custom"
    )
    create_interactions: bool = Field(
        default=False,
        description="Create feature interactions"
    )
    polynomial_degree: int = Field(
        default=1,
        description="Polynomial feature degree (1 = no polynomial features)",
        ge=1,
        le=3
    )


# ============================================================================
# Tool: System Inspection
# ============================================================================

@mcp.tool()
async def system_inspect(ctx: Context[ServerSession, AppContext]) -> dict:
    """
    Inspect system hardware and compute capabilities.
    
    Returns information about CPU, RAM, GPU availability, and
    recommendations for model training.
    """
    import psutil
    
    # CPU info
    cpu_info = {
        "cores": psutil.cpu_count(logical=False) or 1,
        "threads": psutil.cpu_count(logical=True) or 1,
        "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
    }
    
    # Memory info
    mem = psutil.virtual_memory()
    memory_info = {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "usage_percent": mem.percent,
    }
    
    # GPU info
    gpu_info = {"available": False, "devices": [], "total_vram_gb": 0}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
                gpu_info["total_vram_gb"] += props.total_memory / (1024**3)
            gpu_info["total_vram_gb"] = round(gpu_info["total_vram_gb"], 2)
    except ImportError:
        pass
    
    # Disk info
    disk = psutil.disk_usage('/')
    disk_info = {
        "free_gb": round(disk.free / (1024**3), 2),
        "total_gb": round(disk.total / (1024**3), 2),
    }
    
    # Recommendations
    recommendations = {
        "deep_learning_feasible": gpu_info["available"] and gpu_info["total_vram_gb"] >= 4,
        "max_batch_size": _calculate_batch_size(gpu_info, memory_info),
        "recommended_n_jobs": max(1, cpu_info["cores"] - 1),
        "can_use_large_models": memory_info["available_gb"] >= 8,
    }
    
    await ctx.info(f"System inspection complete. GPU available: {gpu_info['available']}")
    
    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "disk": disk_info,
        "recommendations": recommendations,
    }


def _calculate_batch_size(gpu_info: dict, memory_info: dict) -> int:
    """Calculate recommended batch size based on available resources."""
    if gpu_info["available"] and gpu_info["total_vram_gb"] > 0:
        vram = gpu_info["total_vram_gb"]
        if vram >= 24:
            return 64
        elif vram >= 12:
            return 32
        elif vram >= 8:
            return 16
        else:
            return 8
    else:
        ram = memory_info["available_gb"]
        if ram >= 32:
            return 128
        elif ram >= 16:
            return 64
        else:
            return 32


# ============================================================================
# Tool: Plan Strategy
# ============================================================================

@mcp.tool()
async def plan_strategy(
    data_path: str,
    target_column: Optional[str] = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> dict:
    """
    Analyze dataset and create modeling strategy.
    
    Args:
        data_path: Path to the dataset (CSV, Parquet, or directory for images)
        target_column: Optional target column name (auto-detect if None)
    
    Returns:
        Dataset info, data quality assessment, risks, and model recommendations.
    """
    app_ctx = ctx.request_context.lifespan_context
    detector = app_ctx.dataset_detector
    
    await ctx.report_progress(progress=0.1, total=1.0, message="Loading dataset...")
    
    # Detect modality and load data
    modality = detector.detect_modality(data_path)
    dataset_info = detector.analyze_dataset(data_path, target_column)
    
    await ctx.report_progress(progress=0.5, total=1.0, message="Analyzing data quality...")
    
    # Data quality checks
    data_quality = detector.check_data_quality(data_path)
    
    await ctx.report_progress(progress=0.8, total=1.0, message="Generating recommendations...")
    
    # Risk assessment and model recommendations
    risks = detector.identify_risks(dataset_info, data_quality)
    recommended_models = detector.recommend_models(dataset_info, modality)
    preprocessing_steps = detector.suggest_preprocessing(dataset_info, data_quality)
    
    await ctx.info(f"Strategy planned for {modality} dataset with {dataset_info.get('n_samples', 0)} samples")
    
    return {
        "dataset_info": dataset_info,
        "data_quality": data_quality,
        "risks": risks,
        "recommended_models": recommended_models,
        "preprocessing_steps": preprocessing_steps,
    }


# ============================================================================
# Tool: Run EDA
# ============================================================================

@mcp.tool()
async def run_eda(
    data_path: str,
    modality: Optional[str] = None,
    save_plots: bool = True,
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """
    Perform comprehensive exploratory data analysis.
    
    Args:
        data_path: Path to the dataset
        modality: Force specific modality (tabular, timeseries, vision, nlp) or auto-detect
        save_plots: Whether to generate and save visualization plots
    
    Returns:
        Path to the generated Markdown EDA report.
    """
    from src.eda import run_eda_pipeline
    
    app_ctx = ctx.request_context.lifespan_context
    detector = app_ctx.dataset_detector
    version_manager = app_ctx.version_manager
    
    await ctx.report_progress(progress=0.1, total=1.0, message="Detecting data modality...")
    
    # Auto-detect modality if not specified
    if modality is None:
        modality = detector.detect_modality(data_path)
    
    await ctx.info(f"Running EDA for {modality} dataset")
    
    # Run EDA pipeline
    report_path = await run_eda_pipeline(
        data_path=data_path,
        modality=modality,
        save_plots=save_plots,
        version_manager=version_manager,
        progress_callback=lambda p, m: ctx.report_progress(progress=p, total=1.0, message=m),
    )
    
    await ctx.info(f"EDA report generated: {report_path}")
    
    return report_path


# ============================================================================
# Tool: Train Automatic
# ============================================================================

@mcp.tool()
async def train_automatic(
    data_path: str,
    target_column: Optional[str] = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> dict:
    """
    Fully automatic training workflow with user-configured time budget.
    
    The agent will:
    1. Ask for time budget and constraints
    2. Run EDA if first version
    3. Select and train appropriate models
    4. Track experiments and suggest next steps
    
    Args:
        data_path: Path to training data
        target_column: Target column name (auto-detect if None)
    
    Returns:
        Training results including version, models trained, metrics, and recommendations.
    """
    from src.training import run_automatic_training
    
    app_ctx = ctx.request_context.lifespan_context
    constraint_manager = app_ctx.constraint_manager
    version_manager = app_ctx.version_manager
    
    # Elicit time budget configuration from user
    await ctx.info("Requesting training configuration from user...")
    
    result = await ctx.elicit(
        message="Configure your training session (default: 3 hours, all models allowed):",
        schema=TimeBudgetConfig,
    )
    
    if result.action != "accept":
        return {"status": "cancelled", "message": "Training cancelled by user"}
    
    config = result.data
    
    # Set constraints
    constraint_manager.set_time_budget(config.time_budget_hours)
    constraint_manager.set_deep_learning_allowed(config.allow_deep_learning)
    constraint_manager.set_basic_ml_only(config.basic_ml_only)
    
    await ctx.info(
        f"Training configured: {config.time_budget_hours}h budget, "
        f"DL={'enabled' if config.allow_deep_learning else 'disabled'}, "
        f"Basic ML only={'yes' if config.basic_ml_only else 'no'}"
    )
    
    # Run training
    training_result = await run_automatic_training(
        data_path=data_path,
        target_column=target_column,
        version_manager=version_manager,
        constraint_manager=constraint_manager,
        ctx=ctx,
    )
    
    return training_result


# ============================================================================
# Tool: Train with EDA Report
# ============================================================================

@mcp.tool()
async def train_with_eda(
    data_path: str,
    eda_report_path: str,
    target_column: Optional[str] = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> dict:
    """
    Train models using existing EDA report guidance.
    
    Args:
        data_path: Path to training data
        eda_report_path: Path to existing EDA markdown report
        target_column: Target column name
    
    Returns:
        Training results following EDA recommendations.
    """
    from src.training import run_eda_guided_training
    
    app_ctx = ctx.request_context.lifespan_context
    
    await ctx.info(f"Training using EDA report: {eda_report_path}")
    
    return await run_eda_guided_training(
        data_path=data_path,
        eda_report_path=eda_report_path,
        target_column=target_column,
        version_manager=app_ctx.version_manager,
        constraint_manager=app_ctx.constraint_manager,
        ctx=ctx,
    )


# ============================================================================
# Tool: Train Test Mode (Quick Benchmarking)
# ============================================================================

@mcp.tool()
async def train_test_mode(
    data_path: str,
    target_column: Optional[str] = None,
    models_to_test: Optional[list[str]] = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """
    Fast benchmarking of multiple model families.
    
    Uses a small subsample for quick comparison of different model types.
    Generates a model_plan.md with rankings and recommendations.
    
    Args:
        data_path: Path to training data
        target_column: Target column name
        models_to_test: Optional list of model names to benchmark
    
    Returns:
        Path to generated model_plan.md with rankings.
    """
    from src.training import run_benchmark_training
    
    app_ctx = ctx.request_context.lifespan_context
    
    await ctx.info("Running quick model benchmarking...")
    
    return await run_benchmark_training(
        data_path=data_path,
        target_column=target_column,
        models_to_test=models_to_test,
        version_manager=app_ctx.version_manager,
        ctx=ctx,
    )


# ============================================================================
# Tool: Create Features
# ============================================================================

@mcp.tool()
async def create_features(
    data_path: str,
    strategy: Optional[str] = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> dict:
    """
    Generate or update feature engineering pipeline.
    
    Args:
        data_path: Path to raw data
        strategy: Feature engineering strategy (basic, advanced, target_encoding, custom)
    
    Returns:
        Features created, configuration path, and transformed data path.
    """
    from src.features import run_feature_engineering
    
    app_ctx = ctx.request_context.lifespan_context
    
    # Elicit strategy configuration if not provided
    if strategy is None:
        result = await ctx.elicit(
            message="Configure feature engineering strategy:",
            schema=FeatureStrategyConfig,
        )
        
        if result.action != "accept":
            return {"status": "cancelled", "message": "Feature engineering cancelled"}
        
        strategy = result.data.strategy
    
    await ctx.info(f"Creating features with strategy: {strategy}")
    
    return await run_feature_engineering(
        data_path=data_path,
        strategy=strategy,
        version_manager=app_ctx.version_manager,
        ctx=ctx,
    )


# ============================================================================
# Tool: Evaluate Model
# ============================================================================

@mcp.tool()
async def evaluate_model(
    version: str,
    analysis_type: str = "full",
    ctx: Context[ServerSession, AppContext] = None,
) -> dict:
    """
    Deep error analysis on trained model.
    
    Args:
        version: Version to evaluate (e.g., "v0.3")
        analysis_type: "full", "quick", or "failures_only"
    
    Returns:
        Metrics, confusion matrix, error patterns, and recommendations.
    """
    from src.evaluation import run_model_evaluation
    
    app_ctx = ctx.request_context.lifespan_context
    
    await ctx.info(f"Evaluating model version: {version}")
    
    return await run_model_evaluation(
        version=version,
        analysis_type=analysis_type,
        version_manager=app_ctx.version_manager,
        ctx=ctx,
    )


# ============================================================================
# Tool: Create Ensemble
# ============================================================================

@mcp.tool()
async def create_ensemble(
    versions: list[str],
    method: str = "average",
    weights: Optional[list[float]] = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> dict:
    """
    Combine predictions from multiple model versions.
    
    Args:
        versions: List of version IDs to ensemble
        method: "average", "weighted", or "stacking"
        weights: Optional weights for weighted averaging
    
    Returns:
        Ensemble version, component versions, method, and performance metrics.
    """
    from src.ensemble import run_ensemble_creation
    
    app_ctx = ctx.request_context.lifespan_context
    
    await ctx.info(f"Creating {method} ensemble from {len(versions)} versions")
    
    return await run_ensemble_creation(
        versions=versions,
        method=method,
        weights=weights,
        version_manager=app_ctx.version_manager,
        ctx=ctx,
    )


# ============================================================================
# Tool: Generate Submission
# ============================================================================

@mcp.tool()
async def generate_submission(
    version: str,
    test_data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """
    Generate validated submission file.
    
    Args:
        version: Version to use for predictions
        test_data_path: Path to test data (auto-detect if None)
        output_path: Output file path (auto-generate if None)
    
    Returns:
        Path to the generated submission file.
    """
    from src.submission import generate_submission_file
    
    app_ctx = ctx.request_context.lifespan_context
    
    await ctx.info(f"Generating submission for version: {version}")
    
    return await generate_submission_file(
        version=version,
        test_data_path=test_data_path,
        output_path=output_path,
        version_manager=app_ctx.version_manager,
        ctx=ctx,
    )


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
