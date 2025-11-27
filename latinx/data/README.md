# LatinX Data Generators

This module provides dataset generators for various analytical functions useful for testing ML models, particularly for function approximation, uncertainty quantification, and Bayesian methods.

## Available Datasets

### 1. Sine-Cosine (`sine_cosine`)

Generates 1D sinusoidal data with configurable amplitude, frequency, and noise.

```python
from latinx.data.sine_cosine import SineCosineTranslator

translator = SineCosineTranslator(
    amplitude=2.0,
    angle_multiplier=1.0,
    num_samples=100,
    noise_std=0.1,
    seed=42
)
data = translator.generate()  # Returns DataFrame with 't', 'sine', 'cosine', 'x1', 'y1'
```

**Use cases**: Simple 1D regression, time series, periodicity learning

---

### 2. Mexican Hat (`mexican_hat`)

Generates 2D radially symmetric Laplacian of Gaussian (Mexican Hat wavelet) data.

**Equation**: `z(r) = (1 - r²/σ²) * exp(-r²/(2σ²))` where `r = sqrt(x² + y²)`

```python
from latinx.data.mexican_hat import MexicanHatTranslator

translator = MexicanHatTranslator(
    sigma=1.5,           # Scale parameter
    amplitude=2.0,       # Amplitude multiplier
    x_range=(-5, 5),     # X coordinate range
    y_range=(-5, 5),     # Y coordinate range
    grid_size=100,       # Points per axis (100x100=10k points)
    noise_std=0.1,       # Gaussian noise
    seed=42
)

# Flat DataFrame format
data = translator.generate()  # Returns DataFrame: 'x', 'y', 'z', 'z_noisy', 'r'

# Meshgrid format for 3D plotting
X, Y, Z = translator.generate_grid()  # Returns meshgrids for plotting
```

**Features**:
- Radially symmetric
- Single central peak with surrounding trough
- Smooth, differentiable everywhere
- Good for testing 2D function approximation

**Use cases**:
- 2D regression
- Feature detection
- Testing spatial extrapolation
- Radial basis function learning

---

### 3. Bessel Ripple (`bessel_ripple`)

Generates 2D radial wave patterns resembling water droplet ripples using spherical Bessel functions.

**Equations**:
- **Bessel**: `z(r) = A * j₀(k*r)` where `j₀` is spherical Bessel function
- **Simple**: `z(r) = A * sin(k*r) / r` (close approximation)
- **With damping**: `z(r) = A * j₀(k*r) * exp(-α*r)`

```python
from latinx.data.bessel_ripple import BesselRippleTranslator

translator = BesselRippleTranslator(
    k=6.0,                # Wave number (higher = more ripples)
    amplitude=1.0,        # Amplitude multiplier
    damping=0.05,         # Exponential decay (0 = no damping)
    x_range=(-10, 10),    # X coordinate range
    y_range=(-10, 10),    # Y coordinate range
    grid_size=100,        # Points per axis
    noise_std=0.05,       # Gaussian noise
    use_bessel=True,      # True: spherical Bessel, False: sin(kr)/r
    seed=42
)

# Flat DataFrame format
data = translator.generate()  # Returns DataFrame: 'x', 'y', 'z', 'z_noisy', 'r'

# Meshgrid format for 3D plotting
X, Y, Z = translator.generate_grid()  # Returns meshgrids
```

**Features**:
- Radially symmetric oscillating waves
- Decays with distance (like real ripples)
- Two implementations: accurate Bessel vs fast sin(kr)/r
- Optional exponential damping

**Use cases**:
- Complex 2D regression with oscillations
- Testing model behavior on high-frequency functions
- Physical simulation data
- Radial pattern recognition

---

## Common API

All dataset generators follow a consistent API:

### Methods

- **`generate()`** → `pd.DataFrame`
  - Returns flattened data as DataFrame
  - Suitable for ML pipelines, pandas operations

- **`generate_grid()`** → `(X, Y, Z)` meshgrids (2D generators only)
  - Returns numpy meshgrids for 3D plotting
  - Compatible with `matplotlib` surface plots

### Common Parameters

- **`seed`**: Random seed for reproducibility
- **`noise_std`**: Standard deviation of additive Gaussian noise
- **`amplitude`**: Output scaling factor

### DataFrame Columns

**1D datasets** (sine-cosine):
- Input: `t`
- Outputs: `sine`, `cosine`, etc.

**2D datasets** (mexican_hat, bessel_ripple):
- Inputs: `x`, `y`
- Outputs: `z` (clean), `z_noisy` (with noise)
- Derived: `r` (radial distance)

---

## Integration with Bayesian Last Layer

All datasets work seamlessly with the Bayesian Last Layer model:

```python
from latinx.data.bessel_ripple import BesselRippleTranslator
from latinx.models.bayesian_last_layer import BayesianLastLayer
import jax.numpy as jnp

# Generate data
translator = BesselRippleTranslator(grid_size=50, noise_std=0.1, seed=42)
data = translator.generate()

# Prepare inputs/outputs
X = jnp.array(data[['x', 'y']].values)  # 2D inputs
y = jnp.array(data['z_noisy'].values)   # 1D outputs

# Train model
bll = BayesianLastLayer(hidden_dims=(20, 20), sigma=0.1, seed=42)
bll.fit(X, y)

# Predict with uncertainty
y_pred, y_std = bll.predict(X, return_std=True)
```

---

## Mathematical Properties

| Dataset | Dimensionality | Smoothness | Oscillatory | Bounded | Radial Symmetry |
|---------|---------------|------------|-------------|---------|-----------------|
| Sine-Cosine | 1D | C∞ | Yes | Yes | N/A |
| Mexican Hat | 2D | C∞ | No | No | Yes |
| Bessel Ripple | 2D | C∞ | Yes | No | Yes |

---

## Examples

See `examples/` directory for complete usage examples:
- `examples/bayesian_last_layer_example.py` - Using sine-cosine data
- `examples/mexican_hat_example.py` - 2D Mexican Hat function learning
- `examples/bessel_ripple_example.py` - Learning water ripple patterns

---

## Testing

All datasets have comprehensive test coverage in `tests/data/`:
- Parameter validation
- Data generation correctness
- Radial symmetry verification
- Noise addition
- Reproducibility
- Edge cases

Run tests:
```bash
uv run pytest tests/data/ -v
```

---

## References

**Mexican Hat**:
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Wavelet theory and applications

**Bessel Ripple**:
- Arfken, G. B., & Weber, H. J. (2005). *Mathematical Methods for Physicists*. Academic Press.
- Spherical Bessel functions in physics
- Water wave propagation models
