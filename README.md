# ih-coverage

Coverage check and sharpness recommendation for quantitative features.

## Functions

- `check_coverage(data, sharpness, min_per_interval=5, feature_indices=None)` - check coverage for multiple columns
- `discretize(data, sharpness)` - discretize data using given sharpness
- `suggest_sharpness(data, min_per_interval=5)` - suggest optimal sharpness for a single column

## Installation

```bash
pip install git+https://github.com/AlGoldin87/ih-coverage.git
```

## Example

```python
import numpy as np
from ih_coverage import suggest_sharpness

data = np.array([1.2, 2.3, 3.4, 4.5, 5.6], dtype=np.float32)
sharpness = suggest_sharpness(data, min_per_interval=2)
print(f"Recommended sharpness: {sharpness}")
```

## License

MIT
