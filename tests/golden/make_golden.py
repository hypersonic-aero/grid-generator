"""Write the reference coordinates of the golden grids (tests/test_golden.py checks them).

The golden grids are the constant-height grids of the frozen configs next to this
script, and one shock-aligned grid (its shock curve next to it, as *_shock.csv); each
config's body is its *_geometry.toml. Their
grids must not change by accident, so rerun this script only for a deliberate
change, and say why in the commit:

    python tests/golden/make_golden.py

Each config gets a .npz file with its grid shape, the coordinates at evenly strided
(i, j) points, and x and y summed along every grid line (so a change confined to one
line shows up too).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from grid_generator import generate_body_fitted

# --------------------------------------------------
# constants
# --------------------------------------------------
GOLDEN_DIR = Path(__file__).resolve().parent

# about this many sampled points along i and along j (the last line is always added)
SAMPLES_I = 25
SAMPLES_J = 20


# --------------------------------------------------
# public API
# --------------------------------------------------
def sample_indices(n_points: int, n_samples: int) -> NDArray[np.int64]:
    """Evenly strided indices from 0, always including the last one.

    Args:
        n_points: Number of points along the index direction.
        n_samples: About how many indices to pick.

    Returns:
        Increasing indices.
    """

    # stride, then make sure the last line is in
    stride = max(1, n_points // n_samples)
    indices = np.arange(0, n_points, stride)
    if indices[-1] != n_points - 1:
        indices = np.append(indices, n_points - 1)

    return indices


def golden_record(config_path: Path) -> dict[str, NDArray]:
    """Generate the grid of one golden config and pick what the test compares.

    Args:
        config_path: Frozen TOML config.

    Returns:
        Arrays to store: shape, sampled indices and coordinates, line sums.
    """

    # generate the (x, y) plane
    dataset = generate_body_fitted(config_path)
    x = dataset.grid.x[:, :, 0]
    y = dataset.grid.y[:, :, 0]

    # sampled points
    i_index = sample_indices(x.shape[0], SAMPLES_I)
    j_index = sample_indices(x.shape[1], SAMPLES_J)
    sampled = np.ix_(i_index, j_index)

    # build the record: every j-line (fixed i) and every i-line (fixed j) summed
    record = {
        "shape": np.array(x.shape),
        "i_index": i_index,
        "j_index": j_index,
        "x": x[sampled],
        "y": y[sampled],
        "x_sum_over_j": x.sum(axis=1),
        "y_sum_over_j": y.sum(axis=1),
        "x_sum_over_i": x.sum(axis=0),
        "y_sum_over_i": y.sum(axis=0),
    }

    return record


def main() -> None:
    """Write one .npz per golden config."""

    # one record per frozen grid config (the *_geometry.toml files are their bodies)
    config_paths = sorted(
        path for path in GOLDEN_DIR.glob("*.toml") if not path.stem.endswith("_geometry")
    )
    for config_path in config_paths:
        record = golden_record(config_path)
        output_path = config_path.with_suffix(".npz")
        np.savez(output_path, **record)
        print(f"wrote {output_path.name}: grid {record['shape'].tolist()}")


if __name__ == "__main__":
    main()
