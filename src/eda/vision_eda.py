"""Vision EDA - Analysis for image datasets."""

from pathlib import Path
from typing import Any, Optional
from datetime import datetime


class VisionEDA:
    """Exploratory Data Analysis for image datasets."""
    
    def analyze(self, data_path: str) -> dict[str, Any]:
        """
        Perform vision EDA.
        
        Args:
            data_path: Path to image directory
            
        Returns:
            Dictionary with analysis results
        """
        from PIL import Image
        import os
        
        path = Path(data_path)
        results = {}
        
        IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        if path.is_file():
            # Single image
            try:
                img = Image.open(path)
                results["type"] = "single_image"
                results["size"] = img.size
                results["mode"] = img.mode
                results["format"] = img.format
            except Exception as e:
                results["error"] = str(e)
            return results
        
        # Directory of images
        all_images = list(path.rglob('*'))
        images = [f for f in all_images if f.suffix.lower() in IMAGE_EXTENSIONS]
        
        results["type"] = "image_directory"
        results["total_images"] = len(images)
        
        # Check for class structure
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if subdirs:
            class_counts = {}
            for subdir in subdirs:
                class_images = [f for f in subdir.rglob('*') 
                               if f.suffix.lower() in IMAGE_EXTENSIONS]
                class_counts[subdir.name] = len(class_images)
            
            results["n_classes"] = len(class_counts)
            results["class_distribution"] = class_counts
            
            # Check for imbalance
            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                results["is_imbalanced"] = max_count / (min_count + 1) > 3
        
        # Sample image statistics
        if images:
            sample_size = min(100, len(images))
            widths, heights, modes, formats = [], [], [], []
            
            for img_path in images[:sample_size]:
                try:
                    img = Image.open(img_path)
                    widths.append(img.size[0])
                    heights.append(img.size[1])
                    modes.append(img.mode)
                    formats.append(img.format or img_path.suffix)
                except Exception:
                    pass
            
            if widths:
                results["image_stats"] = {
                    "avg_width": sum(widths) // len(widths),
                    "avg_height": sum(heights) // len(heights),
                    "min_width": min(widths),
                    "max_width": max(widths),
                    "min_height": min(heights),
                    "max_height": max(heights),
                    "modes": list(set(modes)),
                    "formats": list(set(formats)),
                    "size_variance": len(set(zip(widths, heights))) > 1,
                }
        
        return results
    
    def generate_plots(self, data_path: str, output_dir: str) -> list[str]:
        """Generate vision dataset visualizations."""
        import matplotlib.pyplot as plt
        from PIL import Image
        import numpy as np
        
        path = Path(data_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plots = []
        IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        if path.is_file():
            return plots
        
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        
        # Class distribution bar chart
        if subdirs:
            class_counts = {}
            for subdir in subdirs:
                class_images = [f for f in subdir.rglob('*') 
                               if f.suffix.lower() in IMAGE_EXTENSIONS]
                class_counts[subdir.name] = len(class_images)
            
            if class_counts:
                fig, ax = plt.subplots(figsize=(10, 6))
                classes = list(class_counts.keys())
                counts = list(class_counts.values())
                ax.bar(classes, counts, edgecolor='black')
                ax.set_title('Class Distribution')
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_path = output_path / 'class_distribution.png'
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                plots.append(str(plot_path))
        
        # Sample images grid
        images = list(path.rglob('*'))
        images = [f for f in images if f.suffix.lower() in IMAGE_EXTENSIONS][:9]
        
        if images:
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            for i, ax in enumerate(axes.flatten()):
                if i < len(images):
                    try:
                        img = Image.open(images[i])
                        ax.imshow(np.array(img))
                        ax.set_title(images[i].parent.name if subdirs else images[i].name[:20])
                    except Exception:
                        ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.axis('off')
            
            plt.suptitle('Sample Images')
            plt.tight_layout()
            plot_path = output_path / 'sample_images.png'
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            plots.append(str(plot_path))
        
        return plots
    
    def generate_report(
        self,
        results: dict[str, Any],
        plots_dir: Optional[Path] = None
    ) -> str:
        """Generate Markdown EDA report for vision dataset."""
        report = []
        
        report.append("# Vision Dataset EDA Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Modality:** Computer Vision\n")
        
        # Overview
        report.append("## 1. Dataset Overview")
        report.append(f"- **Type:** {results.get('type', 'N/A')}")
        report.append(f"- **Total Images:** {results.get('total_images', 'N/A'):,}")
        
        if "n_classes" in results:
            report.append(f"- **Number of Classes:** {results['n_classes']}\n")
        
        # Class distribution
        if "class_distribution" in results:
            report.append("## 2. Class Distribution")
            if results.get("is_imbalanced"):
                report.append("> ⚠️ **Warning:** Class imbalance detected\n")
            report.append("| Class | Count |")
            report.append("|-------|-------|")
            for cls, count in sorted(results["class_distribution"].items(), key=lambda x: -x[1]):
                report.append(f"| {cls} | {count:,} |")
            report.append("")
        
        # Image statistics
        if "image_stats" in results:
            stats = results["image_stats"]
            report.append("## 3. Image Statistics")
            report.append(f"- **Average Size:** {stats.get('avg_width', 'N/A')} x {stats.get('avg_height', 'N/A')}")
            report.append(f"- **Size Range:** {stats.get('min_width', 'N/A')}-{stats.get('max_width', 'N/A')} x {stats.get('min_height', 'N/A')}-{stats.get('max_height', 'N/A')}")
            report.append(f"- **Modes:** {', '.join(stats.get('modes', []))}")
            report.append(f"- **Formats:** {', '.join(stats.get('formats', []))}")
            if stats.get("size_variance"):
                report.append("> ⚠️ Images have varying sizes - resize will be needed\n")
        
        # Plots
        if plots_dir:
            report.append("## 4. Visualizations")
            plots_path = Path(plots_dir)
            for plot_file in plots_path.glob("*.png"):
                report.append(f"\n![{plot_file.stem}]({plot_file})")
        
        # Recommendations
        report.append("\n## 5. Recommendations")
        if results.get("is_imbalanced"):
            report.append("- Consider data augmentation for minority classes")
        if results.get("image_stats", {}).get("size_variance"):
            report.append("- Resize all images to consistent dimensions")
        report.append("- Use transfer learning with EfficientNet or ResNet")
        report.append("- Apply data augmentation (rotation, flip, color jitter)")
        
        return "\n".join(report)
