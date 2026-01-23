"""Time Series EDA - Analysis for time series datasets."""

from pathlib import Path
from typing import Any, Optional
from datetime import datetime


class TimeSeriesEDA:
    """Exploratory Data Analysis for time series data."""
    
    def analyze(self, data_path: str) -> dict[str, Any]:
        """
        Perform time series EDA.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Dictionary with analysis results
        """
        import pandas as pd
        import numpy as np
        
        # Load data
        path = Path(data_path)
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=True)
        
        results = {}
        
        # Basic stats
        results["basic_stats"] = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "columns": df.columns.tolist(),
        }
        
        # Find datetime column
        date_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
            if col.lower() in ['date', 'datetime', 'timestamp', 'time', 'ds']:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_col = col
                    break
                except Exception:
                    pass
        
        if date_col:
            results["time_column"] = date_col
            results["date_range"] = {
                "start": str(df[date_col].min()),
                "end": str(df[date_col].max()),
                "duration_days": (df[date_col].max() - df[date_col].min()).days,
            }
            
            # Frequency detection
            if len(df) > 1:
                time_diffs = df[date_col].diff().dropna()
                most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else None
                if most_common_diff:
                    if most_common_diff <= pd.Timedelta(hours=1):
                        freq = "hourly"
                    elif most_common_diff <= pd.Timedelta(days=1):
                        freq = "daily"
                    elif most_common_diff <= pd.Timedelta(weeks=1):
                        freq = "weekly"
                    else:
                        freq = "monthly or longer"
                    results["estimated_frequency"] = freq
        
        # Target/value column analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]  # Assume first numeric is target
            series = df[value_col].dropna()
            
            results["target_column"] = value_col
            results["target_stats"] = {
                "mean": round(series.mean(), 4),
                "std": round(series.std(), 4),
                "min": round(series.min(), 4),
                "max": round(series.max(), 4),
                "median": round(series.median(), 4),
            }
            
            # Trend detection (simple linear regression)
            if len(series) > 10:
                x = np.arange(len(series))
                slope, _ = np.polyfit(x, series.values, 1)
                if abs(slope) < series.std() * 0.01:
                    trend = "no clear trend"
                elif slope > 0:
                    trend = "upward trend"
                else:
                    trend = "downward trend"
                results["trend"] = trend
            
            # Stationarity (simple check via variance split)
            if len(series) > 20:
                first_half_var = series[:len(series)//2].var()
                second_half_var = series[len(series)//2:].var()
                variance_ratio = max(first_half_var, second_half_var) / (min(first_half_var, second_half_var) + 1e-10)
                results["stationarity_check"] = {
                    "variance_ratio": round(variance_ratio, 2),
                    "likely_stationary": variance_ratio < 2,
                }
        
        # Missing values
        missing = df.isnull().sum()
        results["missing_values"] = {
            col: int(count) for col, count in missing[missing > 0].items()
        }
        
        return results
    
    def generate_plots(self, data_path: str, output_dir: str) -> list[str]:
        """Generate time series visualizations."""
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        
        path = Path(data_path)
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=True)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plots = []
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Find numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]
            series = df[value_col].dropna()
            
            # Time series plot
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(series.values)
            ax.set_title(f'Time Series: {value_col}')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Value')
            plot_path = output_path / 'time_series_plot.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            plots.append(str(plot_path))
            
            # Distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            series.hist(ax=ax, bins=30, edgecolor='black')
            ax.set_title(f'Distribution: {value_col}')
            plot_path = output_path / 'distribution.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            plots.append(str(plot_path))
            
            # Rolling statistics
            if len(series) > 30:
                fig, ax = plt.subplots(figsize=(12, 4))
                window = min(30, len(series) // 5)
                rolling_mean = series.rolling(window=window).mean()
                rolling_std = series.rolling(window=window).std()
                ax.plot(series.values, alpha=0.5, label='Original')
                ax.plot(rolling_mean.values, label=f'Rolling Mean ({window})')
                ax.plot(rolling_std.values, label=f'Rolling Std ({window})')
                ax.legend()
                ax.set_title('Rolling Statistics')
                plot_path = output_path / 'rolling_stats.png'
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                plots.append(str(plot_path))
        
        return plots
    
    def generate_report(
        self,
        results: dict[str, Any],
        plots_dir: Optional[Path] = None
    ) -> str:
        """Generate Markdown EDA report for time series."""
        report = []
        
        report.append("# Time Series EDA Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Modality:** Time Series\n")
        
        # Basic stats
        basic = results.get("basic_stats", {})
        report.append("## 1. Dataset Overview")
        report.append(f"- **Samples:** {basic.get('n_samples', 'N/A'):,}")
        report.append(f"- **Features:** {basic.get('n_features', 'N/A')}\n")
        
        # Time info
        if "time_column" in results:
            report.append("## 2. Time Information")
            report.append(f"- **Time Column:** {results['time_column']}")
            date_range = results.get("date_range", {})
            report.append(f"- **Start:** {date_range.get('start', 'N/A')}")
            report.append(f"- **End:** {date_range.get('end', 'N/A')}")
            report.append(f"- **Duration:** {date_range.get('duration_days', 'N/A')} days")
            if "estimated_frequency" in results:
                report.append(f"- **Frequency:** {results['estimated_frequency']}\n")
        
        # Target stats
        if "target_stats" in results:
            report.append("## 3. Target Variable")
            report.append(f"**Column:** {results.get('target_column', 'N/A')}\n")
            stats = results["target_stats"]
            report.append(f"- Mean: {stats.get('mean', 'N/A')}")
            report.append(f"- Std: {stats.get('std', 'N/A')}")
            report.append(f"- Min: {stats.get('min', 'N/A')}")
            report.append(f"- Max: {stats.get('max', 'N/A')}\n")
        
        # Trend
        if "trend" in results:
            report.append("## 4. Trend Analysis")
            report.append(f"- **Detected:** {results['trend']}\n")
        
        # Stationarity
        if "stationarity_check" in results:
            check = results["stationarity_check"]
            report.append("## 5. Stationarity Check")
            report.append(f"- **Variance Ratio:** {check.get('variance_ratio', 'N/A')}")
            report.append(f"- **Likely Stationary:** {'Yes ✓' if check.get('likely_stationary') else 'No ⚠️'}\n")
        
        # Missing values
        missing = results.get("missing_values", {})
        if missing:
            report.append("## 6. Missing Values")
            for col, count in missing.items():
                report.append(f"- {col}: {count}")
            report.append("")
        
        # Plots
        if plots_dir:
            report.append("## 7. Visualizations")
            plots_path = Path(plots_dir)
            for plot_file in plots_path.glob("*.png"):
                report.append(f"\n![{plot_file.stem}]({plot_file})")
        
        # Recommendations
        report.append("\n## 8. Recommendations")
        if "stationarity_check" in results and not results["stationarity_check"].get("likely_stationary"):
            report.append("- Consider differencing to achieve stationarity")
        report.append("- Try ARIMA, Prophet, or LightGBM with lag features")
        report.append("- Create lag features, rolling statistics, and seasonality indicators")
        
        return "\n".join(report)
