"""Tabular EDA - Comprehensive analysis for tabular datasets."""

from pathlib import Path
from typing import Any, Optional
from datetime import datetime


class TabularEDA:
    """Exploratory Data Analysis for tabular data."""
    
    def analyze(self, data_path: str) -> dict[str, Any]:
        """
        Perform comprehensive tabular EDA.
        
        Args:
            data_path: Path to the CSV/Parquet file
            
        Returns:
            Dictionary with analysis results
        """
        import pandas as pd
        import numpy as np
        
        # Load data
        path = Path(data_path)
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix in {'.xlsx', '.xls'}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        
        results = {}
        
        # Basic stats
        results["basic_stats"] = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "columns": df.columns.tolist(),
        }
        
        # Data types
        results["dtypes"] = df.dtypes.astype(str).to_dict()
        
        # Missing values
        missing = df.isnull().sum()
        results["missing_analysis"] = {
            "total_missing": int(missing.sum()),
            "columns_with_missing": {
                col: {"count": int(count), "percentage": round(count / len(df) * 100, 2)}
                for col, count in missing[missing > 0].items()
            },
        }
        
        # Duplicates
        results["duplicates"] = {
            "count": int(df.duplicated().sum()),
            "percentage": round(df.duplicated().sum() / len(df) * 100, 2),
        }
        
        # Numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            numeric_stats = df[numeric_cols].describe().T
            results["numeric_features"] = {
                "columns": numeric_cols,
                "count": len(numeric_cols),
                "stats": numeric_stats.to_dict(),
            }
            
            # Correlation matrix (top correlations)
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                # Get top correlations
                corr_pairs = []
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        corr_pairs.append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": round(corr.loc[col1, col2], 3),
                        })
                corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                results["top_correlations"] = corr_pairs[:10]
        
        # Categorical features
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            cat_analysis = {}
            for col in cat_cols:
                value_counts = df[col].value_counts()
                cat_analysis[col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_values": value_counts.head(5).to_dict(),
                    "missing": int(df[col].isnull().sum()),
                }
            results["categorical_features"] = {
                "columns": cat_cols,
                "count": len(cat_cols),
                "analysis": cat_analysis,
            }
        
        # Outliers (IQR method for numeric)
        if numeric_cols:
            outliers = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
                if outlier_count > 0:
                    outliers[col] = {
                        "count": int(outlier_count),
                        "percentage": round(outlier_count / len(df) * 100, 2),
                        "lower_bound": round(lower, 3),
                        "upper_bound": round(upper, 3),
                    }
            results["outliers"] = outliers
        
        # Target analysis (if last column is likely target)
        target_col = df.columns[-1]
        if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 20:
            value_counts = df[target_col].value_counts()
            results["target_analysis"] = {
                "column": target_col,
                "type": "categorical",
                "n_classes": int(df[target_col].nunique()),
                "distribution": value_counts.to_dict(),
                "is_imbalanced": value_counts.max() / value_counts.min() > 3,
            }
        else:
            results["target_analysis"] = {
                "column": target_col,
                "type": "continuous",
                "mean": round(df[target_col].mean(), 3),
                "std": round(df[target_col].std(), 3),
                "min": round(df[target_col].min(), 3),
                "max": round(df[target_col].max(), 3),
            }
        
        return results
    
    def generate_plots(self, data_path: str, output_dir: str) -> list[str]:
        """
        Generate visualization plots.
        
        Args:
            data_path: Path to the dataset
            output_dir: Directory to save plots
            
        Returns:
            List of generated plot paths
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        path = Path(data_path)
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plots = []
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Missing values heatmap
        if df.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(df.isnull(), cbar=True, yticklabels=False, ax=ax)
            ax.set_title('Missing Values Heatmap')
            plot_path = output_path / 'missing_values.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            plots.append(str(plot_path))
        
        # Numeric distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:8]  # Limit to 8
        if len(numeric_cols) > 0:
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
            axes = np.array(axes).flatten()
            
            for i, col in enumerate(numeric_cols):
                df[col].hist(ax=axes[i], bins=30, edgecolor='black')
                axes[i].set_title(col)
            
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = output_path / 'numeric_distributions.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            plots.append(str(plot_path))
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            plot_path = output_path / 'correlation_heatmap.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            plots.append(str(plot_path))
        
        # Target distribution
        target_col = df.columns[-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 20:
            df[target_col].value_counts().plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_title(f'Target Distribution: {target_col}')
        else:
            df[target_col].hist(ax=ax, bins=30, edgecolor='black')
            ax.set_title(f'Target Distribution: {target_col}')
        
        plot_path = output_path / 'target_distribution.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plots.append(str(plot_path))
        
        return plots
    
    def generate_report(
        self,
        results: dict[str, Any],
        plots_dir: Optional[Path] = None
    ) -> str:
        """
        Generate Markdown EDA report.
        
        Args:
            results: Analysis results from analyze()
            plots_dir: Optional directory containing plots
            
        Returns:
            Markdown report string
        """
        report = []
        
        # Header
        report.append("# Exploratory Data Analysis Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Modality:** Tabular\n")
        
        # Basic stats
        basic = results.get("basic_stats", {})
        report.append("## 1. Dataset Overview")
        report.append(f"- **Samples:** {basic.get('n_samples', 'N/A'):,}")
        report.append(f"- **Features:** {basic.get('n_features', 'N/A')}")
        report.append(f"- **Memory:** {basic.get('memory_mb', 'N/A')} MB\n")
        
        # Missing values
        missing = results.get("missing_analysis", {})
        report.append("## 2. Missing Values")
        if missing.get("total_missing", 0) == 0:
            report.append("No missing values detected. ✓\n")
        else:
            report.append(f"Total missing values: {missing.get('total_missing', 0):,}\n")
            report.append("| Column | Count | Percentage |")
            report.append("|--------|-------|------------|")
            for col, info in missing.get("columns_with_missing", {}).items():
                report.append(f"| {col} | {info['count']:,} | {info['percentage']:.2f}% |")
            report.append("")
        
        # Duplicates
        dups = results.get("duplicates", {})
        report.append("## 3. Duplicates")
        report.append(f"- **Count:** {dups.get('count', 0):,}")
        report.append(f"- **Percentage:** {dups.get('percentage', 0):.2f}%\n")
        
        # Numeric features
        numeric = results.get("numeric_features", {})
        if numeric:
            report.append("## 4. Numeric Features")
            report.append(f"**Count:** {numeric.get('count', 0)}\n")
            report.append("| Feature | Mean | Std | Min | Max |")
            report.append("|---------|------|-----|-----|-----|")
            stats = numeric.get("stats", {})
            for col in numeric.get("columns", [])[:10]:
                mean = stats.get("mean", {}).get(col, 0)
                std = stats.get("std", {}).get(col, 0)
                min_val = stats.get("min", {}).get(col, 0)
                max_val = stats.get("max", {}).get(col, 0)
                report.append(f"| {col} | {mean:.2f} | {std:.2f} | {min_val:.2f} | {max_val:.2f} |")
            report.append("")
        
        # Categorical features
        categorical = results.get("categorical_features", {})
        if categorical:
            report.append("## 5. Categorical Features")
            report.append(f"**Count:** {categorical.get('count', 0)}\n")
            report.append("| Feature | Unique Values |")
            report.append("|---------|---------------|")
            for col, info in list(categorical.get("analysis", {}).items())[:10]:
                report.append(f"| {col} | {info['unique_values']} |")
            report.append("")
        
        # Top correlations
        correlations = results.get("top_correlations", [])
        if correlations:
            report.append("## 6. Top Correlations")
            report.append("| Feature 1 | Feature 2 | Correlation |")
            report.append("|-----------|-----------|-------------|")
            for corr in correlations[:5]:
                report.append(f"| {corr['feature1']} | {corr['feature2']} | {corr['correlation']:.3f} |")
            report.append("")
        
        # Target analysis
        target = results.get("target_analysis", {})
        if target:
            report.append("## 7. Target Variable Analysis")
            report.append(f"**Column:** {target.get('column', 'N/A')}")
            report.append(f"**Type:** {target.get('type', 'N/A')}\n")
            
            if target.get("type") == "categorical":
                if target.get("is_imbalanced"):
                    report.append("> ⚠️ **Warning:** Class imbalance detected\n")
                report.append("| Class | Count |")
                report.append("|-------|-------|")
                for cls, count in list(target.get("distribution", {}).items())[:10]:
                    report.append(f"| {cls} | {count:,} |")
            else:
                report.append(f"- Mean: {target.get('mean', 'N/A')}")
                report.append(f"- Std: {target.get('std', 'N/A')}")
                report.append(f"- Range: [{target.get('min', 'N/A')}, {target.get('max', 'N/A')}]")
            report.append("")
        
        # Outliers
        outliers = results.get("outliers", {})
        if outliers:
            report.append("## 8. Outliers (IQR Method)")
            report.append("| Feature | Count | Percentage |")
            report.append("|---------|-------|------------|")
            for col, info in list(outliers.items())[:10]:
                report.append(f"| {col} | {info['count']:,} | {info['percentage']:.2f}% |")
            report.append("")
        
        # Plots
        if plots_dir:
            report.append("## 9. Visualizations")
            plots_path = Path(plots_dir)
            for plot_file in plots_path.glob("*.png"):
                report.append(f"\n![{plot_file.stem}]({plot_file})")
        
        # Recommendations
        report.append("\n## 10. Recommendations")
        
        recommendations = []
        if missing.get("total_missing", 0) > 0:
            recommendations.append("- Handle missing values (imputation or removal)")
        if dups.get("count", 0) > 0:
            recommendations.append("- Consider removing duplicate rows")
        if outliers:
            recommendations.append("- Review and handle outliers")
        if target.get("is_imbalanced"):
            recommendations.append("- Address class imbalance (SMOTE, class weights)")
        if categorical:
            recommendations.append("- Encode categorical features")
        if numeric:
            recommendations.append("- Scale numerical features")
        
        if recommendations:
            for rec in recommendations:
                report.append(rec)
        else:
            report.append("- Data appears clean and ready for modeling")
        
        return "\n".join(report)
