"""Golden grids: the grids of frozen configs (constant-height, and one shock-aligned) must not change.

tests/golden holds the configs, the bodies they point at (*_geometry.toml), and their
reference coordinates (written by tests/golden/make_golden.py). A deliberate change to these grids reruns that script
and says why in the commit; any other difference is a regression.

The comparison allows round-off (1e-12 of the grid's extent): the sines, cosines and
powers behind the grid may differ in the last bit between platforms.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grid_generator import generate_body_fitted

# --------------------------------------------------
# constants
# --------------------------------------------------
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
# the grid configs (each body's geometry file, *_geometry.toml, sits next to its config)
GOLDEN_CONFIGS = sorted(
    path for path in GOLDEN_DIR.glob("*.toml") if not path.stem.endswith("_geometry")
)

# allowed difference, relative to the grid's extent
TOLERANCE = 1.0e-12


# --------------------------------------------------
# tests
# --------------------------------------------------
def test_golden_configs_exist() -> None:
    """Every frozen config has its reference data (guards against a packaging slip)."""

    # check
    assert GOLDEN_CONFIGS
    for config_path in GOLDEN_CONFIGS:
        assert config_path.with_suffix(".npz").exists(), config_path.name


@pytest.mark.parametrize("config_path", GOLDEN_CONFIGS, ids=lambda path: path.stem)
def test_golden_grid_is_unchanged(config_path: Path) -> None:
    """The grid has the stored shape, sampled points and line sums, to round-off."""

    # read the reference
    reference = np.load(config_path.with_suffix(".npz"))

    # generate the (x, y) plane
    dataset = generate_body_fitted(config_path)
    x = dataset.grid.x[:, :, 0]
    y = dataset.grid.y[:, :, 0]

    # check: same shape
    assert list(x.shape) == reference["shape"].tolist()

    # check: sampled points, reporting the largest point difference
    extent = max(float(np.ptp(x)), float(np.ptp(y)))
    sampled = np.ix_(reference["i_index"], reference["j_index"])
    difference = np.hypot(x[sampled] - reference["x"], y[sampled] - reference["y"])
    worst = np.unravel_index(int(np.argmax(difference)), difference.shape)
    worst_i = int(reference["i_index"][worst[0]])
    worst_j = int(reference["j_index"][worst[1]])
    assert difference.max() <= TOLERANCE * extent, (
        f"{config_path.name}: points moved by up to {difference.max():.3e} m "
        f"(at i = {worst_i}, j = {worst_j}); if deliberate, rerun "
        "tests/golden/make_golden.py and say why in the commit"
    )

    # check: every grid line's sums (catches a change confined to one line)
    nj = x.shape[1]
    ni = x.shape[0]
    line_sums = {
        "x_sum_over_j": (x.sum(axis=1), nj),
        "y_sum_over_j": (y.sum(axis=1), nj),
        "x_sum_over_i": (x.sum(axis=0), ni),
        "y_sum_over_i": (y.sum(axis=0), ni),
    }
    for key, (sums, n_terms) in line_sums.items():
        np.testing.assert_allclose(
            sums, reference[key], rtol=0.0, atol=TOLERANCE * extent * n_terms, err_msg=key
        )
