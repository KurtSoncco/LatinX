# LatinX

**LatinX** — Collaborative research & ML experiments on Bayesian Last Layer

A modern Python project for machine learning research focused on Bayesian methods, built with JAX-first approach and PyTorch support.

## Features

- 🔬 **Research-focused**: Built for collaborative ML experimentation
- 🚀 **JAX-first**: Leverages JAX for high-performance numerical computing
- 🔥 **PyTorch support**: Includes PyTorch (CPU default) for flexible model development
- 📊 **Rich visualization**: matplotlib, seaborn, and interactive plotly
- 📓 **Jupyter-ready**: Integrated JupyterLab for interactive development

## Prerequisites

- Python 3.11 or higher (< 4.0)
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver

### Installing uv

If you don't have `uv` installed:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: via pip
pip install uv
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/KurtSoncco/LatinX.git
cd LatinX
```

### 2. Install dependencies

**Basic installation (core dependencies only):**

```bash
uv sync
```

**Development installation (includes testing, linting, and type checking):**

```bash
uv sync --extra dev
```

This will:
- Create a virtual environment in `.venv/`
- Install all dependencies specified in `pyproject.toml`
- Set up the project in editable mode

### 3. Activate the virtual environment

```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

## GPU Support
Not implemmented yet. 

## Project Structure

```
LatinX/
├── code/               # Source code
│   ├── data/          # Data processing utilities
│   └── models/        # Model implementations
├── docs/              # Documentation
├── notebooks/         # Jupyter notebooks for experiments
│   └── Sine_LRU.ipynb
├── results/           # Experiment results and outputs
├── scripts/           # Utility scripts
├── tests/             # Unit tests
├── pyproject.toml     # Project configuration and dependencies
├── LICENSE            # MIT License
└── README.md          # This file
```

## Usage

### Running Jupyter Lab

```bash
jupyter lab
```

### Running tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=code tests/
```

### Code formatting

```bash
# Format code with black
black code/ tests/

# Sort imports
isort code/ tests/
```

### Type checking

```bash
mypy code/
```

## Dependencies

### Core Dependencies

- **Scientific Computing**: numpy, pandas, scipy
- **Machine Learning**: JAX, PyTorch
- **Visualization**: matplotlib, plotly, seaborn
- **Interactive Development**: JupyterLab

### Development Dependencies

- **Testing**: pytest, pytest-cov
- **Code Quality**: black, isort, mypy
- **Git Hooks**: pre-commit

## Troubleshooting

### CUDA Package Download Timeout

If you encounter timeout errors with NVIDIA CUDA packages (e.g., `nvidia-nvshmem-cu12`):

```bash
# Increase the timeout (default: 30s)
UV_HTTP_TIMEOUT=120 uv sync
```

Or switch to CPU-only PyTorch by modifying `pyproject.toml`:

```toml
dependencies = [
  # ... other deps ...
  "torch>=2.2,<2.3; platform_system != 'Darwin'",  # CPU only
]
```

### JAX Version Check

To verify your installed JAX version:

```bash
python -c "import jax; print(jax.__version__)"
```

## Contributing

1. Install development dependencies: `uv sync --extra dev`
2. Set up pre-commit hooks: `pre-commit install`
3. Create a feature branch: `git checkout -b feature-name`
4. Make your changes and ensure tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **KurtSoncco** - [kurtwal98@berkeley.edu](mailto:kurtwal98@berkeley.edu)

## Acknowledgments

Built with modern Python tooling:
- [uv](https://github.com/astral-sh/uv) - Ultra-fast Python package installer
- [JAX](https://github.com/google/jax) - High-performance numerical computing
- [PyTorch](https://pytorch.org/) - Deep learning framework
