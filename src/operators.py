import numpy as np
from scipy.sparse import csr_matrix

from .graph import GraphData


def _validate_values(
    values: np.ndarray,
    graph: GraphData,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.shape != (graph.n_nodes,):
        raise ValueError("values must have shape (n_nodes,).")

    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain finite values.")

    return values


def _validate_p(p: float) -> float:
    if not np.isscalar(p):
        raise TypeError("p must be a scalar.")

    p = float(p)

    if p == np.inf:
        return p

    if not np.isfinite(p) or p < 1:
        raise ValueError("p must satisfy p >= 1 or p = infinity.")

    return p


def _directional_difference(
    values: np.ndarray,
    graph: GraphData,
    direction: str,
) -> csr_matrix:
    values = _validate_values(values, graph)
    W = graph.weights

    rows = np.repeat(
        np.arange(graph.n_nodes),
        np.diff(W.indptr),
    )

    delta = values[W.indices] - values[rows]

    if direction == "plus":
        delta = np.maximum(delta, 0.0)
    elif direction == "minus":
        delta = np.maximum(-delta, 0.0)
    else:
        raise ValueError("direction must be plus or minus.")

    result = csr_matrix(
        (
            np.sqrt(W.data) * delta,
            W.indices.copy(),
            W.indptr.copy(),
        ),
        shape=W.shape,
    )
    result.eliminate_zeros()

    return result


def difference_plus(values, graph):
    return _directional_difference(values, graph, "plus")


def difference_minus(values, graph):
    return _directional_difference(values, graph, "minus")


def gradient_plus(values, graph):
    return difference_plus(values, graph)


def gradient_minus(values, graph):
    return difference_minus(values, graph)


def gradient_norm(
    values: np.ndarray,
    graph: GraphData,
    p: float = 2.0,
    direction: str = "plus",
) -> np.ndarray:
    p = _validate_p(p)
    gradient = _directional_difference(values, graph, direction)

    norms = np.zeros(graph.n_nodes)

    if p == np.inf:
        for u in range(graph.n_nodes):
            start = gradient.indptr[u]
            end = gradient.indptr[u + 1]

            if start < end:
                norms[u] = gradient.data[start:end].max()

    else:
        rows = np.repeat(
            np.arange(graph.n_nodes),
            np.diff(gradient.indptr),
        )
        sums = np.bincount(
            rows,
            weights=gradient.data ** p,
            minlength=graph.n_nodes,
        )
        norms = sums ** (1.0 / p)

    if not np.all(np.isfinite(norms)):
        raise FloatingPointError("Gradient norms overflowed.")

    return norms


def p_laplacian(
    values: np.ndarray,
    graph: GraphData,
    p: float = 2.0,
) -> np.ndarray:
    values = _validate_values(values, graph)
    p = _validate_p(p)

    if p == np.inf:
        raise ValueError("Use infinity_laplacian for p = infinity.")

    W = graph.weights

    rows = np.repeat(
        np.arange(graph.n_nodes),
        np.diff(W.indptr),
    )
    delta = values[W.indices] - values[rows]

    terms = (
        W.data ** (p / 2.0)
        * np.sign(delta)
        * np.abs(delta) ** (p - 1.0)
    )

    return np.bincount(
        rows,
        weights=terms,
        minlength=graph.n_nodes,
    )


def infinity_laplacian(
    values: np.ndarray,
    graph: GraphData,
) -> np.ndarray:
    plus = gradient_norm(values, graph, np.inf, "plus")
    minus = gradient_norm(values, graph, np.inf, "minus")

    return 0.5 * (plus - minus)


def mean_curvature(
    values: np.ndarray,
    graph: GraphData,
) -> np.ndarray:
    values = _validate_values(values, graph)
    W = graph.weights

    rows = np.repeat(
        np.arange(graph.n_nodes),
        np.diff(W.indptr),
    )
    delta = values[W.indices] - values[rows]
    roots = np.sqrt(W.data)

    # Toutain uses sign(0) = 1 for this curvature.
    signs = np.where(delta >= 0.0, 1.0, -1.0)

    numerator = np.bincount(
        rows,
        weights=roots * signs,
        minlength=graph.n_nodes,
    )
    denominator = np.bincount(
        rows,
        weights=roots,
        minlength=graph.n_nodes,
    )

    curvature = np.zeros(graph.n_nodes)
    valid = denominator > 0

    curvature[valid] = numerator[valid] / denominator[valid]

    return curvature


def normalized_p_laplacian(
    values: np.ndarray,
    graph: GraphData,
    p: float = 2.0,
) -> np.ndarray:
    values = _validate_values(values, graph)
    p = _validate_p(p)

    if np.any(graph.weights.data > 1.0):
        raise ValueError(
            "This statistical normalization requires weights <= 1."
        )

    mean = values.copy()
    median = values.copy()

    for u in range(graph.n_nodes):
        neighbors, weights = graph.neighbors(u)

        if neighbors.size == 0:
            continue

        mean[u] = np.dot(weights, values[neighbors]) / weights.sum()

        differences = np.sqrt(weights) * (
            values[neighbors] - values[u]
        )
        median[u] = values[u] + np.median(differences)

    if p <= 2:
        mean_coefficient = 2.0 * (p - 1.0) / p
        median_coefficient = (2.0 - p) / p

        return (
            mean_coefficient * (mean - values)
            + median_coefficient * (median - values)
        )

    midrange_difference = infinity_laplacian(values, graph)

    if p == np.inf:
        return midrange_difference

    return (
        (2.0 / p) * (mean - values)
        + ((p - 2.0) / p) * midrange_difference
    )

