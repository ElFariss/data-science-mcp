"""NLP EDA - Analysis for text datasets."""

from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from collections import Counter


class NLPEDA:
    """Exploratory Data Analysis for NLP datasets."""
    
    def analyze(self, data_path: str) -> dict[str, Any]:
        """
        Perform NLP EDA.
        
        Args:
            data_path: Path to text dataset (CSV with text column)
            
        Returns:
            Dictionary with analysis results
        """
        import pandas as pd
        import re
        
        path = Path(data_path)
        results = {}
        
        # Load data
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix in {'.txt'}:
            with open(path, 'r') as f:
                texts = f.readlines()
            df = pd.DataFrame({'text': texts})
        else:
            df = pd.read_csv(path)
        
        results["n_samples"] = len(df)
        
        # Find text column
        text_candidates = ['text', 'content', 'description', 'review', 'comment', 
                          'title', 'body', 'message', 'sentence', 'document']
        text_col = None
        for col in text_candidates:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            # Find longest string column
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].astype(str).str.len().mean() > 50:
                    text_col = col
                    break
        
        if text_col is None:
            text_col = df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else df.columns[0]
        
        results["text_column"] = text_col
        texts = df[text_col].astype(str)
        
        # Text length statistics
        lengths = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        results["length_stats"] = {
            "avg_chars": round(lengths.mean(), 1),
            "min_chars": int(lengths.min()),
            "max_chars": int(lengths.max()),
            "avg_words": round(word_counts.mean(), 1),
            "min_words": int(word_counts.min()),
            "max_words": int(word_counts.max()),
        }
        
        # Vocabulary analysis
        all_words = ' '.join(texts.values).lower()
        all_words = re.sub(r'[^\w\s]', '', all_words)
        words = all_words.split()
        
        word_freq = Counter(words)
        results["vocabulary"] = {
            "total_words": len(words),
            "unique_words": len(word_freq),
            "top_words": dict(word_freq.most_common(20)),
        }
        
        # Missing/empty texts
        empty_count = (texts.str.strip() == '').sum() + texts.isnull().sum()
        results["empty_texts"] = {
            "count": int(empty_count),
            "percentage": round(empty_count / len(df) * 100, 2),
        }
        
        # Check for target column (classification)
        target_candidates = ['label', 'target', 'class', 'category', 'sentiment']
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None and len(df.columns) > 1:
            # Last column that's not the text column
            for col in reversed(df.columns.tolist()):
                if col != text_col:
                    target_col = col
                    break
        
        if target_col:
            target = df[target_col]
            results["target_column"] = target_col
            results["target_distribution"] = target.value_counts().to_dict()
            results["n_classes"] = target.nunique()
        
        return results
    
    def generate_plots(self, data_path: str, output_dir: str) -> list[str]:
        """Generate NLP visualizations."""
        import pandas as pd
        import matplotlib.pyplot as plt
        import re
        from collections import Counter
        
        path = Path(data_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plots = []
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Load data
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        
        # Find text column
        text_col = None
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].astype(str).str.len().mean() > 30:
                text_col = col
                break
        
        if text_col is None:
            return plots
        
        texts = df[text_col].astype(str)
        
        # Word count distribution
        word_counts = texts.str.split().str.len()
        fig, ax = plt.subplots(figsize=(10, 4))
        word_counts.hist(ax=ax, bins=50, edgecolor='black')
        ax.set_title('Word Count Distribution')
        ax.set_xlabel('Number of Words')
        ax.set_ylabel('Frequency')
        plot_path = output_path / 'word_count_dist.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plots.append(str(plot_path))
        
        # Top words bar chart
        all_words = ' '.join(texts.values).lower()
        all_words = re.sub(r'[^\w\s]', '', all_words)
        words = all_words.split()
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'could', 'should', 'may', 'might', 'must', 'shall', 'can',
                    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                    'and', 'or', 'not', 'this', 'that', 'it', 'as', 'but', 'if',
                    'they', 'them', 'their', 'i', 'you', 'he', 'she', 'we', 'me'}
        words_filtered = [w for w in words if w not in stopwords and len(w) > 2]
        word_freq = Counter(words_filtered)
        
        top_words = word_freq.most_common(15)
        if top_words:
            fig, ax = plt.subplots(figsize=(10, 6))
            words_list, counts = zip(*top_words)
            ax.barh(words_list, counts, edgecolor='black')
            ax.set_title('Top 15 Words (excluding stopwords)')
            ax.set_xlabel('Frequency')
            ax.invert_yaxis()
            plot_path = output_path / 'top_words.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            plots.append(str(plot_path))
        
        # Target distribution if exists
        target_cols = ['label', 'target', 'class', 'category', 'sentiment']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            fig, ax = plt.subplots(figsize=(8, 6))
            df[target_col].value_counts().plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_title(f'Target Distribution: {target_col}')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
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
        """Generate Markdown EDA report for NLP dataset."""
        report = []
        
        report.append("# NLP Dataset EDA Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Modality:** Natural Language Processing\n")
        
        # Overview
        report.append("## 1. Dataset Overview")
        report.append(f"- **Samples:** {results.get('n_samples', 'N/A'):,}")
        report.append(f"- **Text Column:** {results.get('text_column', 'N/A')}\n")
        
        # Text length stats
        length_stats = results.get("length_stats", {})
        report.append("## 2. Text Length Statistics")
        report.append(f"- **Average Characters:** {length_stats.get('avg_chars', 'N/A')}")
        report.append(f"- **Character Range:** {length_stats.get('min_chars', 'N/A')} - {length_stats.get('max_chars', 'N/A')}")
        report.append(f"- **Average Words:** {length_stats.get('avg_words', 'N/A')}")
        report.append(f"- **Word Range:** {length_stats.get('min_words', 'N/A')} - {length_stats.get('max_words', 'N/A')}\n")
        
        # Vocabulary
        vocab = results.get("vocabulary", {})
        report.append("## 3. Vocabulary Analysis")
        report.append(f"- **Total Words:** {vocab.get('total_words', 'N/A'):,}")
        report.append(f"- **Unique Words:** {vocab.get('unique_words', 'N/A'):,}")
        
        top_words = vocab.get("top_words", {})
        if top_words:
            report.append("\n**Top 10 Words:**")
            report.append("| Word | Count |")
            report.append("|------|-------|")
            for word, count in list(top_words.items())[:10]:
                report.append(f"| {word} | {count:,} |")
            report.append("")
        
        # Empty texts
        empty = results.get("empty_texts", {})
        if empty.get("count", 0) > 0:
            report.append("## 4. Empty/Missing Texts")
            report.append(f"- **Count:** {empty.get('count', 0):,}")
            report.append(f"- **Percentage:** {empty.get('percentage', 0):.2f}%\n")
        
        # Target distribution
        if "target_column" in results:
            report.append("## 5. Target Distribution")
            report.append(f"**Column:** {results['target_column']}")
            report.append(f"**Classes:** {results.get('n_classes', 'N/A')}\n")
            report.append("| Class | Count |")
            report.append("|-------|-------|")
            for cls, count in list(results.get("target_distribution", {}).items())[:10]:
                report.append(f"| {cls} | {count:,} |")
            report.append("")
        
        # Plots
        if plots_dir:
            report.append("## 6. Visualizations")
            plots_path = Path(plots_dir)
            for plot_file in plots_path.glob("*.png"):
                report.append(f"\n![{plot_file.stem}]({plot_file})")
        
        # Recommendations
        report.append("\n## 7. Recommendations")
        report.append("- Preprocess: lowercasing, removing punctuation, handling special characters")
        report.append("- Consider TF-IDF baseline or BERT-based models")
        report.append("- For short texts, try DistilBERT or ALBERT")
        report.append("- For longer texts, consider sliding window or truncation strategies")
        
        return "\n".join(report)
