import numpy as np
from scipy.sparse import csr_matrix

from graph import GraphData

def _validate_values(
    values: np.ndarray,
    graph: GraphData,
) -> np.ndarray:
    """
    Validate a scalar function defined on graph vertices.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be a one-dimensional array."
        )

    return values


def _directional_difference(
    values: np.ndarray,
    graph: GraphData,
    direction: str,
) -> csr_matrix:
    """
    Internal implementation of directional differences.
    """

    values = _validate_values(
        values,
        graph,
    )

    W = graph.weights

    degrees = np.diff(W.indptr)

    rows = np.repeat(
        np.arange(W.shape[0]),
        degrees,
    )

    cols = W.indices

    delta = (
        values[cols]
        - values[rows]
    )

    if direction == "plus":

        directional = np.maximum(
            delta,
            0.0,
        )

    elif direction == "minus":

        directional = np.maximum(
            -delta,
            0.0,
        )

    else:
        raise ValueError(
            "direction must be 'plus' or 'minus'."
        )

    data = (
        np.sqrt(W.data)
        * directional
    )

    result = csr_matrix(
        (
            data,
            W.indices.copy(),
            W.indptr.copy(),
        ),
        shape=W.shape,
    )

    result.eliminate_zeros()

    return result

def difference_plus(
    values: np.ndarray,
    graph: GraphData,
) -> csr_matrix:
    """
    Positive/external directional weighted difference.
    """

    return _directional_difference(
        values,
        graph,
        direction="plus",
    )

def difference_minus(
    values: np.ndarray,
    graph: GraphData,
) -> csr_matrix:
    """
    Negative/internal directional weighted difference.
    """

    return _directional_difference(
        values,
        graph,
        direction="minus",
    )

def gradient_plus(
    values: np.ndarray,
    graph: GraphData,
) -> csr_matrix:
    """
    Compute the external directional gradient.

    For each vertex u,

        grad_w^+(f)(u)
            = (d_w^+(f)(u, v))_v.

    The gradients of all vertices are stored
    row-wise in a sparse matrix.
    """

    return difference_plus(
        values,
        graph,
    )

def gradient_minus(
    values: np.ndarray,
    graph: GraphData,
) -> csr_matrix:
    """
    Compute the internal directional gradient.

    For each vertex u,

        grad_w^-(f)(u)
            = (d_w^-(f)(u, v))_v.

    The gradients of all vertices are stored
    row-wise in a sparse matrix.
    """

    return difference_minus(
        values,
        graph,
    )

def _directional_gradient(
    values: np.ndarray,
    graph: GraphData,
    direction: str,
) -> csr_matrix:
    """
    Return the selected directional gradient.
    """

    if direction == "plus":
        return gradient_plus(
            values,
            graph,
        )

    if direction == "minus":
        return gradient_minus(
            values,
            graph,
        )

    raise ValueError(
        "direction must be 'plus' or 'minus'."
    )

def gradient_norm(
    values: np.ndarray,
    graph: GraphData,
    p: float = 2.0,
    direction: str = "plus",
) -> np.ndarray:
    """
    Compute the Lp norm of a directional graph gradient.

    Parameters
    ----------
    values : np.ndarray
        Scalar values defined on graph vertices.

    graph : GraphData
        Weighted graph.

    p : float
        Norm exponent. Must satisfy p >= 1,
        or be np.inf.

    direction : {"plus", "minus"}
        Directional gradient to use.

    Returns
    -------
    np.ndarray
        One gradient norm value per graph vertex.
    """

    if not (
        np.isscalar(p)
        or p is np.inf
    ):
        raise TypeError(
            "p must be a scalar."
        )

    if not (
        np.isinf(p)
        or (
            np.isfinite(p)
            and p >= 1.0
        )
    ):
        raise ValueError(
            "p must satisfy p >= 1 "
            "or p = infinity."
        )

    gradient = _directional_gradient(
        values,
        graph,
        direction,
    )

    n_nodes = gradient.shape[0]

    if np.isinf(p):

        norms = np.zeros(
            n_nodes,
            dtype=float,
        )

        for u in range(n_nodes):

            start = gradient.indptr[u]
            end = gradient.indptr[u + 1]

            if start < end:
                norms[u] = np.max(np.abs(gradient.data[start:end]))

        return norms

    powered = gradient.copy()

    powered.data = (np.abs(powered.data) ** p)

    sums = np.asarray(powered.sum(axis=1)).ravel()

    return sums ** (1.0 / p)

def p_laplacian(
    values: np.ndarray,
    graph: GraphData,
    p: float = 2.0,
) -> np.ndarray:
    """
    Compute Toutain's anisotropic graph p-Laplacian.

    For each vertex u,

        Delta^a_{w,p} f(u)
            = sum_{v ~ u}
              w_uv^(p/2)
              |f(v) - f(u)|^(p-2)
              (f(v) - f(u)).

    The implementation uses the equivalent form

        sign(delta) * |delta|^(p-1)

    with sign(0) = 0.

    Parameters
    ----------
    values : np.ndarray
        Scalar function defined on graph vertices.

    graph : GraphData
        Weighted undirected graph.

    p : float
        Exponent satisfying 1 <= p < infinity.

    Returns
    -------
    np.ndarray
        p-Laplacian value at every graph vertex.
    """

    values = _validate_values(
        values,
        graph,
    )

    if not np.isscalar(p):
        raise TypeError(
            "p must be a scalar."
        )

    if (
        not np.isfinite(p)
        or p < 1.0
    ):
        raise ValueError(
            "p must satisfy 1 <= p < infinity."
        )

    W = graph.weights

    n_nodes = W.shape[0]

    degrees = np.diff(W.indptr)

    rows = np.repeat(
        np.arange(n_nodes),
        degrees,
    )

    cols = W.indices

    if W.nnz == 0:
        return np.zeros(
            n_nodes,
            dtype=float,
        )

    delta = (
        values[cols]
        - values[rows]
    )

    nonlinear_difference = (
        np.sign(delta)
        * np.abs(delta) ** (p - 1.0)
    )

    terms = (
        W.data ** (p / 2.0)
        * nonlinear_difference
    )

    result = np.bincount(
        rows,
        weights=terms,
        minlength=n_nodes,
    )

    return result

def infinity_laplacian(
    values: np.ndarray,
    graph: GraphData,
) -> np.ndarray:
    """
    Compute the graph infinity-Laplacian used by Toutain.

    For each vertex u,

        Delta_{w,inf} f(u)
            = 1/2 * (
                ||grad_w^+ f(u)||_inf
                - ||grad_w^- f(u)||_inf
            ).

    Parameters
    ----------
    values : np.ndarray
        Scalar function defined on graph vertices.

    graph : GraphData
        Weighted undirected graph.

    Returns
    -------
    np.ndarray
        Infinity-Laplacian value at every graph vertex.
    """

    values = _validate_values(
        values,
        graph,
    )

    plus_norm = gradient_norm(
        values,
        graph,
        p=np.inf,
        direction="plus",
    )

    minus_norm = gradient_norm(
        values,
        graph,
        p=np.inf,
        direction="minus",
    )

    return 0.5 * (
        plus_norm
        - minus_norm
    )

def mean_curvature(
    values: np.ndarray,
    graph: GraphData,
) -> np.ndarray:
    """
    Compute the graph mean curvature used by Toutain.

    For every non-isolated vertex u,

        kappa_w(u, f)
            =
            sum_{v ~ u}
                sqrt(w_uv) * sign(f(v) - f(u))
            /
            sum_{v ~ u}
                sqrt(w_uv),

    with Toutain's convention

        sign(x) = 1   if x >= 0,
                 -1  if x < 0.

    For isolated vertices, the curvature is defined here as 0.

    Parameters
    ----------
    values : np.ndarray
        Scalar function defined on graph vertices.

    graph : GraphData
        Weighted undirected graph.

    Returns
    -------
    np.ndarray
        Mean curvature at every graph vertex.
    """

    values = _validate_values(
        values,
        graph,
    )

    W = graph.weights
    n_nodes = W.shape[0]

    if W.nnz == 0:
        return np.zeros(
            n_nodes,
            dtype=float,
        )

    degrees = np.diff(W.indptr)

    rows = np.repeat(
        np.arange(n_nodes),
        degrees,
    )

    cols = W.indices

    delta = (
        values[cols]
        - values[rows]
    )

    # Important:
    # Toutain defines sign(0) = +1.

    signs = np.where(
        delta >= 0.0,
        1.0,
        -1.0,
    )

    sqrt_weights = np.sqrt(
        W.data
    )

    numerator = np.bincount(
        rows,
        weights=sqrt_weights * signs,
        minlength=n_nodes,
    )

    denominator = np.bincount(
        rows,
        weights=sqrt_weights,
        minlength=n_nodes,
    )

    curvature = np.zeros(
        n_nodes,
        dtype=float,
    )

    non_isolated = (denominator > 0.0)

    curvature[non_isolated] = (
        numerator[non_isolated]
        / denominator[non_isolated]
    )

    return curvature
