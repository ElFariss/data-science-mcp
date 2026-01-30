# AI Data Science Agent MCP Server

An autonomous AI agent for end-to-end machine learning workflows, built using the Model Context Protocol (MCP).

## Overview

This MCP server enables AI assistants (GitHub Copilot, Antigravity, Claude) to perform comprehensive data science tasks:

- **CSV Dataset Reading & Analysis** - Load and analyze datasets automatically
- **Automated EDA** - Exploratory Data Analysis with Markdown reports
- **Model Training** - 100+ model architectures across 4 domains
- **Feature Engineering** - Compare different strategies
- **Time-Budget Constraints** - Configurable training time limits (default 3 hours)
- **Multi-Model & Hybrid Support** - Ensemble and stacking methods
- **Basic ML Mode** - Restrict to simpler models when requested

## Supported Domains

| Domain | Models |
|--------|--------|
| **Tabular** | Linear, SVM, Trees, XGBoost, LightGBM, CatBoost, TabNet, etc. |
| **Time Series** | ARIMA, Prophet, LSTM, Transformer, Hybrid models |
| **Computer Vision** | ResNet, EfficientNet, ViT, YOLO, U-Net |
| **NLP** | TF-IDF, BERT, RoBERTa, GPT, T5 |

## Installation

```bash
# Using uv (recommended)
cd data-science-mcp
uv pip install -e .

# Or with optional deep learning support
uv pip install -e ".[deep-learning]"
```

## Usage

### Running the Server

```bash
# Stdio transport (for Copilot/Antigravity)
python -m src.server

# Or with uv
uv run ds-mcp
```

### MCP Tools Available

| Tool | Description |
|------|-------------|
| `system_inspect` | Inspect hardware (CPU, RAM, GPU) |
| `ping` | Server health check |
| `plan_strategy` | Analyze dataset and propose strategy |
| `run_eda` | Generate comprehensive EDA report |
| `train_automatic` | Fully automatic training with time budget |
| `train_with_eda` | Train using EDA report guidance |
| `train_test_mode` | Quick model benchmarking |
| `create_features` | Run feature engineering |
| `evaluate_model` | Deep error analysis |
| `create_ensemble` | Combine multiple models |
| `generate_submission` | Create submission file |

### Example Workflow

```python
# The AI agent should always pass 'output_dir' for absolute path resolution:

# 1. Check system capabilities
await system_inspect()

# 2. Analyze dataset and plan strategy
await plan_strategy(
    data_path="data/train.csv",
    output_dir="/abs/path/to/project"
)

# 3. Run EDA
await run_eda(
    data_path="data/train.csv",
    output_dir="/abs/path/to/project"
)

# 4. Train automatically (explicit 30 min timeout)
await train_automatic(
    data_path="data/train.csv",
    output_dir="/abs/path/to/project",
    timeout_minutes=30
)

# 5. Evaluate results
await evaluate_model(
    version="v0.1",
    output_dir="/abs/path/to/project"
)

# 6. Generate submission
await generate_submission(
    version="v0.1",
    output_dir="/abs/path/to/project"
)
```

## Time Budget Configuration

When calling `train_automatic`, the server uses MCP elicitation to ask:

- **Time Budget**: How many hours for training (default: 3)
- **Allow Deep Learning**: Enable GPU-based models
- **Basic ML Only**: Restrict to simpler models

Alternatively, agents can pass `timeout_minutes` directly to skip elicitation.

## Path Resolution

To ensure reliability, all tools accept an `output_dir` parameter. This directs the server where to write artifacts (reports, models, submissions). The server guarantees that all returned paths are **absolute**, resolving them relative to the provided `output_dir` (or the server's CWD if not provided).

## Project Structure

```
data-science-mcp/
├── src/
│   ├── server.py           # Main MCP server
│   ├── core/               # Version, Constraint, Dataset managers
│   │   ├── path_utils.py   # Path resolution utilities
│   │   └── ...
│   ├── eda/                # EDA modules (tabular, timeseries, vision, nlp)
│   ├── features/           # Feature engineering
│   ├── models/             # Model registry, selector, multi-model runner
│   ├── training/           # Training pipeline
│   ├── evaluation/         # Error analysis
│   ├── ensemble/           # Ensemble methods
│   └── submission/         # Submission generator
├── experiments/            # Experiment tracking
├── reports/                # EDA reports
└── pyproject.toml
```

## License

MIT
