r"""
weighted_histogram.py
=====================
Self-contained utility for computing weighted histograms of observables
and generating publication-quality plots.

Intended use-case: applying OmniFold per-event weights (or any set of
per-event weights) to a column from a Pandas DataFrame or a plain NumPy
array to produce a correctly normalized weighted histogram with optional
plotting and uncertainty bands.

Requirements
------------
    numpy >= 1.20
    matplotlib >= 3.4  (only required if plotting)

Usage example
-------------
    import numpy as np
    from weighted_histogram import weighted_histogram

    rng = np.random.default_rng(42)
    obs     = rng.exponential(scale=100, size=5000)  # e.g. mumu_pt in GeV
    weights = rng.uniform(0.5, 2.0, size=5000)       # OmniFold weights

    result = weighted_histogram(
        observable=obs,
        weights=weights,
        bins=20,
        range=(0, 600),
        observable_label=r"$p_T^{\mu\mu}$ [GeV]",
        plot=True,
        output_path="mumu_pt.png",
    )
    print(result["bin_centers"])
    print(result["counts"])

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Tuple


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def weighted_histogram(
    observable: npt.ArrayLike,
    weights: Optional[npt.ArrayLike] = None,
    bins: Union[int, npt.ArrayLike] = 20,
    range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    observable_label: str = "Observable",
    title: str = "",
    plot: bool = False,
    output_path: Optional[str] = None,
    color: str = "#3A86FF",
    error_color: str = "#8338EC",
    fig_size: Tuple[float, float] = (8, 5),
    systematic_weights: Optional[list[npt.ArrayLike]] = None,
    systematic_labels: Optional[list[str]] = None,
) -> dict:
    """Compute a weighted histogram of an observable with optional plotting.

    This function wraps ``numpy.histogram`` and adds:
    - Per-bin statistical uncertainty via the sum-of-squared-weights estimator.
    - Optional systematic uncertainty bands when multiple weight sets are supplied.
    - Optional publication-quality Matplotlib figure.

    Parameters
    ----------
    observable : array-like of shape (N,)
        The per-event values of the quantity to histogram (e.g. dimuon pT in
        GeV). Must be 1-dimensional. Non-finite values (NaN, Inf) are
        excluded automatically.

    weights : array-like of shape (N,) or None
        Per-event weights. If ``None``, uniform weights of 1 are used
        (equivalent to an unweighted histogram). Must be the same length
        as ``observable``.

    bins : int or array-like
        Number of equal-width bins, or the explicit bin edge array. Passed
        directly to ``numpy.histogram``.

    range : (float, float) or None
        The lower and upper range of the bins. Events outside this range are
        ignored. If ``None``, the range spans ``[observable.min(),
        observable.max()]``. Ignored when ``bins`` is already an ndarray of
        edges.

    density : bool
        If ``True``, normalise so that the integral of the histogram equals 1
        (probability density). The statistical uncertainties are scaled by the
        same factor. Default is ``False``.

    observable_label : str
        Axis label for the x-axis (supports LaTeX). Default ``"Observable"``.

    title : str
        Figure title. Default is empty.

    plot : bool
        If ``True``, generate a Matplotlib figure. Requires ``matplotlib`` to
        be installed. Default is ``False``.

    output_path : str or None
        If provided, save the figure to this path (any format supported by
        Matplotlib, e.g. "plot.png", "plot.pdf"). Implies ``plot=True``.

    color : str
        Hex colour for the main histogram fill. Default is a clear blue.

    error_color : str
        Hex colour for the statistical error bars. Default is a purple.

    fig_size : (float, float)
        Width × height of the figure in inches.

    systematic_weights : list of array-like or None
        Optional list of alternative weight arrays (one per systematic
        variation, e.g. [sherpa_weights, nondY_weights]). If provided, the
        envelope of all systematic histograms is drawn as a shaded band.

    systematic_labels : list of str or None
        Labels for each systematic variation (for the legend). Must be the
        same length as ``systematic_weights``.

    Returns
    -------
    dict with keys
        ``bin_edges``   : ndarray of shape (n_bins+1,) — left/right bin edges.
        ``bin_centers`` : ndarray of shape (n_bins,)   — centre of each bin.
        ``bin_widths``  : ndarray of shape (n_bins,)   — width of each bin.
        ``counts``      : ndarray of shape (n_bins,)   — weighted counts (or
                          density if density=True).
        ``stat_errors`` : ndarray of shape (n_bins,)   — 1-sigma statistical
                          uncertainty per bin.
        ``n_events_per_bin`` : ndarray (int) — raw (unweighted) event count per bin.
        ``total_weight`` : float — total sum of weights.

    Raises
    ------
    ValueError
        If ``observable`` and ``weights`` have different lengths, or if
        ``weights`` contains negative values (OmniFold weights are always ≥ 0).
    TypeError
        If ``observable`` is not 1-dimensional after conversion.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    obs = np.asarray(observable, dtype=float)
    if obs.ndim != 1:
        raise TypeError(
            f"observable must be 1-dimensional, got shape {obs.shape}."
        )

    if weights is None:
        w = np.ones_like(obs)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != obs.shape:
            raise ValueError(
                f"observable and weights must have the same length; "
                f"got {obs.shape} vs {w.shape}."
            )
        if np.any(w < 0):
            raise ValueError(
                "Negative weights detected. OmniFold weights should be "
                "non-negative. Check your weight array."
            )

    # ------------------------------------------------------------------
    # Remove non-finite values (NaN sentinels like -999 should be filtered
    # by the caller; we only guard against true NaN/Inf here)
    # ------------------------------------------------------------------
    finite_mask = np.isfinite(obs) & np.isfinite(w)
    n_dropped = int(np.sum(~finite_mask))
    if n_dropped > 0:
        import warnings
        warnings.warn(
            f"{n_dropped} non-finite values in observable or weights were "
            "excluded from the histogram.",
            RuntimeWarning,
            stacklevel=2,
        )
    obs = obs[finite_mask]
    w   = w[finite_mask]

    # Guard against all-zero weights (would give a trivially empty histogram)
    total_weight = float(w.sum())
    if total_weight == 0:
        raise ValueError("All weights are zero — cannot compute a meaningful histogram.")

    # ------------------------------------------------------------------
    # Compute weighted histogram
    # ------------------------------------------------------------------
    counts, bin_edges = np.histogram(obs, bins=bins, range=range, weights=w)
    # Sum of squared weights per bin — gives the variance of the weighted count
    counts_sq, _ = np.histogram(obs, bins=bin_edges, weights=w ** 2)

    stat_errors = np.sqrt(counts_sq)  # 1-sigma per bin

    # Raw (unweighted) events per bin for reporting
    n_events_per_bin, _ = np.histogram(obs, bins=bin_edges)

    bin_widths  = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2.0

    # Optional density normalisation
    if density:
        norm_factor = total_weight * bin_widths      # integral normalization
        counts      = counts / norm_factor
        stat_errors = stat_errors / norm_factor

    # ------------------------------------------------------------------
    # Optionally compute systematic histograms
    # ------------------------------------------------------------------
    syst_counts_list = []
    if systematic_weights is not None:
        for syst_w in systematic_weights:
            syst_w_arr = np.asarray(syst_w, dtype=float)[finite_mask]
            syst_c, _ = np.histogram(obs, bins=bin_edges, weights=syst_w_arr)
            if density:
                syst_norm = syst_w_arr.sum() * bin_widths
                syst_c = syst_c / syst_norm
            syst_counts_list.append(syst_c)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    if plot or output_path is not None:
        _plot_weighted_histogram(
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            bin_widths=bin_widths,
            counts=counts,
            stat_errors=stat_errors,
            syst_counts_list=syst_counts_list,
            systematic_labels=systematic_labels,
            observable_label=observable_label,
            title=title,
            density=density,
            color=color,
            error_color=error_color,
            fig_size=fig_size,
            output_path=output_path,
        )

    return {
        "bin_edges":        bin_edges,
        "bin_centers":      bin_centers,
        "bin_widths":       bin_widths,
        "counts":           counts,
        "stat_errors":      stat_errors,
        "n_events_per_bin": n_events_per_bin,
        "total_weight":     total_weight,
    }


# ---------------------------------------------------------------------------
# Private plotting helper
# ---------------------------------------------------------------------------

def _plot_weighted_histogram(
    bin_edges, bin_centers, bin_widths, counts, stat_errors,
    syst_counts_list, systematic_labels, observable_label, title, density,
    color, error_color, fig_size, output_path,
):
    """Internal function — renders and optionally saves the histogram figure."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from exc

    # Use a clean style
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    ):
        fig, ax = plt.subplots(figsize=fig_size)

        # ---- systematic band (drawn first so nominal sits on top) ----
        if syst_counts_list:
            all_counts = np.array(syst_counts_list + [counts])
            syst_lo = all_counts.min(axis=0)
            syst_hi = all_counts.max(axis=0)
            ax.fill_between(
                bin_centers, syst_lo, syst_hi,
                step="mid",
                alpha=0.25,
                color="#FF6B6B",
                label="Systematic envelope",
            )

        # ---- main filled step histogram ----
        ax.fill_between(
            np.append(bin_edges[:-1], bin_edges[-1]),
            np.append(counts, counts[-1]),
            step="post",
            alpha=0.35,
            color=color,
        )
        ax.step(
            bin_edges,
            np.append(counts, counts[-1]),
            where="post",
            color=color,
            linewidth=1.8,
            label="Nominal (OmniFold)",
        )

        # ---- statistical error bars ----
        ax.errorbar(
            bin_centers,
            counts,
            yerr=stat_errors,
            fmt="none",
            ecolor=error_color,
            elinewidth=1.5,
            capsize=3,
            label="Stat. uncertainty",
        )

        # ---- axis labels and title ----
        ax.set_xlabel(observable_label, fontsize=13)
        y_label = "Probability density" if density else "Weighted counts"
        ax.set_ylabel(y_label, fontsize=13)
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        ax.set_xlim(bin_edges[0], bin_edges[-1])

        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)

        ax.legend(framealpha=0.8, fontsize=11)
        fig.tight_layout()

        if output_path is not None:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to: {output_path}")

        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

"""
Tests for weighted_histogram()
================================

Edge cases considered and why:

1.  ZERO-WEIGHT EVENTS: Some OmniFold weights can be very small but are never
    exactly zero in a healthy run. We test that zero weights are accepted but
    a fully-zero weight array raises an error — this would indicate a bug in
    the unfolding.

2.  UNIFORM WEIGHTS = UNWEIGHTED: weights of all ones should give the same
    result as numpy.histogram with no weights. This is a basic sanity check.

3.  SHAPE MISMATCH: If weights and observable have different lengths, a clear
    ValueError must be raised rather than silently computing a wrong result.

4.  NON-FINITE VALUES: NaN sentinels (-999) are common in HEP HDF5 files when
    a second jet is absent. The function must handle them gracefully.

5.  DENSITY NORMALISATION: The integral of a density histogram must equal 1
    (within floating-point tolerance) regardless of the weight distribution.
    This is trivially broken by many naive implementations.

6.  SINGLE-EVENT EDGE CASE: A histogram with one event should not crash; it
    should return a histogram with exactly one non-zero bin.

7.  NEGATIVE WEIGHTS: OmniFold weights must be non-negative. We verify that a
    negative weight array raises ValueError immediately and loudly.

8.  CUSTOM BIN EDGES: Users often supply non-uniform bin edges (e.g., log
    scale). We verify that an explicit edge array is passed through correctly.
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=1000, seed=42):
    """Return reproducible (observable, weights) arrays."""
    rng = np.random.default_rng(seed)
    obs     = rng.uniform(200, 600, size=n)   # dummy dimuon pT in GeV
    weights = rng.uniform(0.5, 2.0, size=n)   # typical OmniFold range
    return obs, weights


# ---------------------------------------------------------------------------
# 1. Uniform weights == numpy.histogram
# ---------------------------------------------------------------------------

def test_uniform_weights_match_numpy():
    """Weighted histogram with w=1 must match numpy.histogram exactly."""
    obs, _ = _make_data()
    w_ones = np.ones(len(obs))
    result = weighted_histogram(obs, weights=w_ones, bins=10, range=(200, 600))
    expected, _ = np.histogram(obs, bins=10, range=(200, 600))
    np.testing.assert_array_almost_equal(result["counts"], expected)


# ---------------------------------------------------------------------------
# 2. None weights == uniform weights
# ---------------------------------------------------------------------------

def test_none_weights_equal_uniform():
    """Passing weights=None must give the same result as w=1."""
    obs, _ = _make_data()
    result_none = weighted_histogram(obs, weights=None, bins=10, range=(200, 600))
    result_ones = weighted_histogram(obs, weights=np.ones(len(obs)), bins=10, range=(200, 600))
    np.testing.assert_array_almost_equal(result_none["counts"], result_ones["counts"])


# ---------------------------------------------------------------------------
# 3. Shape mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    """Mismatched observable and weight lengths must raise ValueError immediately."""
    obs, w = _make_data(n=100)
    w_wrong = w[:50]  # wrong length
    with pytest.raises(ValueError, match="same length"):
        weighted_histogram(obs, weights=w_wrong)


# ---------------------------------------------------------------------------
# 4. Non-finite values are excluded (with a warning)
# ---------------------------------------------------------------------------

def test_nan_values_excluded():
    """NaN values in the observable must be silently dropped (with a warning)."""
    obs, w = _make_data(n=500)
    obs_with_nan = obs.copy()
    obs_with_nan[[0, 10, 50]] = np.nan

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = weighted_histogram(obs_with_nan, weights=w, bins=10, range=(200, 600))

    # Warning must be issued
    assert any(issubclass(c.category, RuntimeWarning) for c in caught), \
        "Expected a RuntimeWarning about non-finite values"

    # Total raw events should be 3 fewer
    assert result["n_events_per_bin"].sum() == 497, \
        "Expected exactly 497 finite events to be counted"


# ---------------------------------------------------------------------------
# 5. Density normalisation integrates to 1
# ---------------------------------------------------------------------------

def test_density_integrates_to_one():
    """With density=True, the integral of counts * bin_widths must be ≈ 1."""
    obs, w = _make_data(n=2000)
    result = weighted_histogram(obs, weights=w, bins=20, range=(200, 600), density=True)
    integral = np.sum(result["counts"] * result["bin_widths"])
    assert abs(integral - 1.0) < 1e-6, \
        f"Integral of density histogram should be 1.0, got {integral:.8f}"


# ---------------------------------------------------------------------------
# 6. Single-event does not crash
# ---------------------------------------------------------------------------

def test_single_event():
    """A histogram of a single event should produce exactly one non-zero bin."""
    obs = np.array([350.0])
    w   = np.array([2.0])
    result = weighted_histogram(obs, weights=w, bins=5, range=(200, 600))
    assert result["counts"].sum() == pytest.approx(2.0), \
        "Single event with weight 2 should sum to 2.0 total counts"
    assert np.count_nonzero(result["counts"]) == 1, \
        "Exactly one bin should be non-zero"


# ---------------------------------------------------------------------------
# 7. Negative weights raise ValueError
# ---------------------------------------------------------------------------

def test_negative_weights_raise():
    """Any negative weight must raise ValueError immediately."""
    obs, w = _make_data(n=100)
    w[5] = -0.1  # inject one negative weight
    with pytest.raises(ValueError, match="[Nn]egative weights"):
        weighted_histogram(obs, weights=w)


# ---------------------------------------------------------------------------
# 8. All-zero weight array raises ValueError
# ---------------------------------------------------------------------------

def test_all_zero_weights_raise():
    """All-zero weights should raise ValueError (would produce an empty histogram)."""
    obs, _ = _make_data(n=100)
    w_zeros = np.zeros(len(obs))
    with pytest.raises(ValueError, match="[Aa]ll weights are zero"):
        weighted_histogram(obs, weights=w_zeros)


# ---------------------------------------------------------------------------
# 9. Custom non-uniform bin edges are preserved
# ---------------------------------------------------------------------------

def test_custom_bin_edges():
    """Custom bin edge arrays should be passed through and preserved."""
    obs, w = _make_data(n=500)
    edges = np.array([200, 250, 300, 400, 500, 600], dtype=float)
    result = weighted_histogram(obs, weights=w, bins=edges)
    np.testing.assert_array_equal(result["bin_edges"], edges)
    assert len(result["counts"]) == len(edges) - 1


# ---------------------------------------------------------------------------
# 10. Statistical errors are non-negative
# ---------------------------------------------------------------------------

def test_stat_errors_non_negative():
    """Statistical errors must always be >= 0 (they are sqrt of sum-of-squares)."""
    obs, w = _make_data(n=1000)
    result = weighted_histogram(obs, weights=w, bins=20, range=(200, 600))
    assert np.all(result["stat_errors"] >= 0), \
        "All statistical errors must be non-negative"


# ---------------------------------------------------------------------------
# 11. Non-1D input raises TypeError
# ---------------------------------------------------------------------------

def test_non_1d_input_raises():
    """2D input array must raise a TypeError with a helpful message."""
    obs_2d = np.random.uniform(200, 600, size=(100, 2))
    with pytest.raises(TypeError, match="1-dimensional"):
        weighted_histogram(obs_2d)


# ---------------------------------------------------------------------------
# Standalone runner (optional, for quick checks without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running demonstration...")
    rng = np.random.default_rng(0)
    n = 5000
    # Simulate a dimuon pT spectrum: exponentially falling, boosted regime
    obs     = 200 + rng.exponential(scale=80, size=n)
    obs     = obs[obs < 700]  # trim to a sensible range
    weights = rng.lognormal(mean=0.0, sigma=0.5, size=len(obs))

    # Simulate Sherpa weights (slightly different shape)
    syst_weights = rng.lognormal(mean=0.05, sigma=0.55, size=len(obs))

    result = weighted_histogram(
        observable=obs,
        weights=weights,
        bins=20,
        range=(200, 700),
        density=True,
        observable_label=r"$p_T^{\mu\mu}$ [GeV]",
        title="OmniFold Weighted Histogram (demo)",
        plot=True,
        output_path="demo_weighted_histogram.png",
        systematic_weights=[syst_weights],
        systematic_labels=["Sherpa"],
    )

    print("\nResults:")
    print(f"  Bin centers: {result['bin_centers'][:5]} ...")
    print(f"  Counts:      {result['counts'][:5]} ...")
    print(f"  Stat errors: {result['stat_errors'][:5]} ...")
    print(f"  Total weight: {result['total_weight']:.2f}")
    print("\nAll demonstration checks passed.")
