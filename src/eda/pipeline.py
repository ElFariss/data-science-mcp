"""EDA Pipeline - Routes to appropriate EDA module based on modality."""

from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Awaitable

from src.core.version_manager import VersionManager
from src.core.path_utils import get_output_dir, ensure_absolute_return


async def run_eda_pipeline(
    data_path: str,
    modality: str,
    save_plots: bool = True,
    version_manager: Optional[VersionManager] = None,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None,
) -> str:
    """
    Run EDA pipeline for the specified modality.
    
    Args:
        data_path: Path to the dataset
        modality: Data modality (tabular, timeseries, vision, nlp)
        save_plots: Whether to generate plots
        version_manager: Optional version manager for saving reports
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to generated Markdown report
    """
    from src.eda.tabular_eda import TabularEDA
    from src.eda.timeseries_eda import TimeSeriesEDA
    from src.eda.vision_eda import VisionEDA
    from src.eda.nlp_eda import NLPEDA
    
    # Select EDA module based on modality
    eda_modules = {
        "tabular": TabularEDA,
        "timeseries": TimeSeriesEDA,
        "vision": VisionEDA,
        "nlp": NLPEDA,
    }
    
    eda_class = eda_modules.get(modality, TabularEDA)
    eda = eda_class()
    
    if progress_callback:
        await progress_callback(0.2, f"Running {modality} EDA...")
    
    # Run analysis
    results = eda.analyze(data_path)
    
    if progress_callback:
        await progress_callback(0.6, "Generating plots...")
    
    # Generate plots if requested
    plots_dir = None
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use get_output_dir for absolute path
        base_plots_dir = get_output_dir(output_dir, "reports/plots")
        plots_dir = base_plots_dir / f"eda_{timestamp}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        eda.generate_plots(data_path, str(plots_dir))
    
    if progress_callback:
        await progress_callback(0.8, "Generating report...")
    
    # Generate Markdown report
    report = eda.generate_report(results, plots_dir)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use get_output_dir for absolute path
    reports_dir = get_output_dir(output_dir, "reports")
    
    report_path = reports_dir / f"eda_{modality}_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Copy to version folder if available
    if version_manager:
        latest_version = version_manager.get_latest_version()
        if latest_version:
            version_manager.copy_to_version(
                str(report_path),
                latest_version,
                f"eda_report.md"
            )
    
    if progress_callback:
        await progress_callback(1.0, "EDA complete")
    
    return ensure_absolute_return(report_path)
