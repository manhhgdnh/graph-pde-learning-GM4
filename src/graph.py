from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors

from .data import extract_patches


@dataclass
class GraphData:
    weights: csr_matrix

    def __post_init__(self) -> None:
        if not issparse(self.weights):
            raise TypeError("weights must be a scipy sparse matrix.")

        if np.iscomplexobj(self.weights.data):
            raise TypeError("Graph weights must be real.")

        W = self.weights.tocsr(copy=True).astype(float)
        W.sum_duplicates()
        W.eliminate_zeros()
        W.sort_indices()

        if W.shape[0] != W.shape[1] or W.shape[0] == 0:
            raise ValueError("weights must be a non-empty square matrix.")

        if not np.all(np.isfinite(W.data)):
            raise ValueError("Graph weights must be finite.")

        if np.any(W.data <= 0):
            raise ValueError("Stored weights must be strictly positive.")

        if np.any(W.diagonal() != 0):
            raise ValueError("The graph diagonal must be zero.")

        if (W - W.T).nnz != 0:
            raise ValueError("Graph weights must be symmetric.")

        self.weights = W

    @property
    def n_nodes(self) -> int:
        return self.weights.shape[0]

    @property
    def n_edges(self) -> int:
        return self.weights.nnz // 2

    def _check_vertex(self, u: int) -> None:
        if isinstance(u, (bool, np.bool_)):
            raise TypeError("Vertex indices must be integers.")

        if not isinstance(u, (int, np.integer)):
            raise TypeError("Vertex indices must be integers.")

        if u < 0 or u >= self.n_nodes:
            raise IndexError("Vertex index is out of range.")

    def neighbors(self, u: int) -> tuple[np.ndarray, np.ndarray]:
        self._check_vertex(u)

        start = self.weights.indptr[u]
        end = self.weights.indptr[u + 1]

        return (
            self.weights.indices[start:end],
            self.weights.data[start:end],
        )

    def degree(self, u: int) -> int:
        neighbors, _ = self.neighbors(u)
        return neighbors.size

    def weighted_degree(self, u: int) -> float:
        _, weights = self.neighbors(u)
        return float(weights.sum())


def _check_integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")

    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")


def _point_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a non-empty feature matrix.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values.")

    return X


def _make_graph(rows, cols, n_nodes: int) -> GraphData:
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)

    W = csr_matrix(
        (np.ones(rows.size), (rows, cols)),
        shape=(n_nodes, n_nodes),
    )
    W.sum_duplicates()
    W.data[:] = 1.0

    W = W.maximum(W.T)
    W.setdiag(0)
    W.eliminate_zeros()

    return GraphData(W)


def build_knn_graph(
    X: np.ndarray,
    k: int = 10,
) -> GraphData:
    X = _point_features(X)
    _check_integer(k, "k")

    n_samples = X.shape[0]

    if k >= n_samples:
        raise ValueError("k must be smaller than the number of samples.")

    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
    )
    model.fit(X)

    raw_neighbors = model.kneighbors(
        X,
        return_distance=False,
    )

    neighbors = np.empty((n_samples, k), dtype=int)

    for u in range(n_samples):
        candidates = raw_neighbors[u]
        candidates = candidates[candidates != u]
        neighbors[u] = candidates[:k]

    rows = np.repeat(np.arange(n_samples), k)
    cols = neighbors.ravel()

    return _make_graph(rows, cols, n_samples)


def build_epsilon_graph(
    X: np.ndarray,
    epsilon: float,
) -> GraphData:
    X = _point_features(X)

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")

    model = NearestNeighbors(
        radius=epsilon,
        metric="euclidean",
    )
    model.fit(X)

    neighborhoods = model.radius_neighbors(
        X,
        return_distance=False,
    )

    rows = []
    cols = []

    for u, candidates in enumerate(neighborhoods):
        candidates = candidates[candidates != u]
        rows.extend([u] * candidates.size)
        cols.extend(candidates.tolist())

    return _make_graph(rows, cols, X.shape[0])


def build_grid_graph(
    image: np.ndarray,
    connectivity: int = 4,
) -> GraphData:
    image = np.asarray(image)

    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty grayscale or color image.")

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    height, width = image.shape[:2]

    offsets = [(0, 1), (1, 0)]

    if connectivity == 8:
        offsets.extend([(1, 1), (1, -1)])

    rows = []
    cols = []

    for i in range(height):
        for j in range(width):
            u = i * width + j

            for di, dj in offsets:
                ni = i + di
                nj = j + dj

                if 0 <= ni < height and 0 <= nj < width:
                    v = ni * width + nj
                    rows.append(u)
                    cols.append(v)

    return _make_graph(rows, cols, height * width)


def _prepare_features(
    features: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    F = np.asarray(features, dtype=float)

    if F.ndim == 1 and F.size == n_nodes:
        F = F.reshape(n_nodes, 1)

    elif F.ndim == 2:
        if F.shape[0] == n_nodes:
            pass
        elif F.size == n_nodes:
            F = F.reshape(n_nodes, 1)
        else:
            raise ValueError("features do not match the graph.")

    elif F.ndim == 3 and F.shape[0] * F.shape[1] == n_nodes:
        F = F.reshape(n_nodes, F.shape[2])

    else:
        raise ValueError("features do not match the graph.")

    if F.shape[1] == 0 or not np.all(np.isfinite(F)):
        raise ValueError("features must be non-empty and finite.")

    return F


def _edge_distances(
    features: np.ndarray,
    graph: GraphData,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = graph.weights

    rows = np.repeat(
        np.arange(graph.n_nodes),
        np.diff(W.indptr),
    )
    cols = W.indices

    differences = features[rows] - features[cols]

    if metric == "euclidean":
        distances = np.sqrt(np.sum(differences ** 2, axis=1))
    elif metric == "manhattan":
        distances = np.sum(np.abs(differences), axis=1)
    elif metric == "chebyshev":
        distances = np.max(np.abs(differences), axis=1)
    else:
        raise ValueError("Unknown distance metric.")

    return rows, cols, distances


def _local_scales(
    features: np.ndarray,
    m: int,
    metric: str,
) -> np.ndarray:
    _check_integer(m, "local_scale_m")

    if m >= features.shape[0]:
        raise ValueError("local_scale_m must be smaller than n_nodes.")

    model = NearestNeighbors(
        n_neighbors=m + 1,
        metric=metric,
    )
    model.fit(features)

    distances, indices = model.kneighbors(features)
    sigma = np.empty(features.shape[0])

    for u in range(features.shape[0]):
        candidates = distances[u][indices[u] != u]
        sigma[u] = candidates[m - 1]

    if np.any(sigma <= 0):
        raise ValueError(
            "Local scales are zero. Check duplicates or increase local_scale_m."
        )

    return sigma


def compute_weights(
    graph: GraphData,
    features: np.ndarray | None = None,
    *,
    mode: str = "constant",
    metric: str = "euclidean",
    sigma: float | None = None,
    local_scale_m: int = 7,
) -> GraphData:
    W = graph.weights.copy()

    if mode == "constant":
        W.data[:] = 1.0
        return GraphData(W)

    if features is None:
        raise ValueError("features are required for Gaussian weights.")

    F = _prepare_features(features, graph.n_nodes)
    rows, cols, distances = _edge_distances(F, graph, metric)

    if mode == "gaussian_global":
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive.")

        denominator = sigma ** 2

    elif mode == "gaussian_local":
        scales = _local_scales(F, local_scale_m, metric)
        denominator = scales[rows] * scales[cols]

    else:
        raise ValueError("Unknown weighting mode.")

    W.data = np.exp(-(distances ** 2) / denominator)

    # Keep the prescribed topology when Gaussian weights underflow.
    W.data = np.maximum(W.data, np.finfo(float).tiny)

    return GraphData(W)


def build_patch_graph(
    image: np.ndarray,
    patch_size: int = 5,
    search_window_size: int = 51,
    k: int = 10,
    patches: np.ndarray | None = None,
) -> GraphData:
    image = np.asarray(image)

    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty grayscale or color image.")

    _check_integer(search_window_size, "search_window_size", 3)
    _check_integer(k, "k")

    if search_window_size % 2 == 0:
        raise ValueError("search_window_size must be odd.")

    height, width = image.shape[:2]
    n_nodes = height * width

    if patches is None:
        patches = extract_patches(image, patch_size)
    else:
        patches = np.asarray(patches, dtype=float)

    if patches.ndim != 2 or patches.shape[0] != n_nodes:
        raise ValueError("patches must contain one feature vector per pixel.")

    if patches.shape[1] == 0 or not np.all(np.isfinite(patches)):
        raise ValueError("patches must be non-empty and finite.")

    radius = search_window_size // 2
    rows = []
    cols = []

    for i in range(height):
        for j in range(width):
            u = i * width + j

            candidate_rows = np.arange(
                max(0, i - radius),
                min(height, i + radius + 1),
            )
            candidate_cols = np.arange(
                max(0, j - radius),
                min(width, j + radius + 1),
            )

            grid_i, grid_j = np.meshgrid(
                candidate_rows,
                candidate_cols,
                indexing="ij",
            )

            candidates = (grid_i * width + grid_j).ravel()
            candidates = candidates[candidates != u]

            if candidates.size == 0:
                continue

            differences = patches[candidates] - patches[u]
            distances = np.sum(differences ** 2, axis=1)

            order = np.argsort(distances, kind="stable")
            selected = candidates[order[:min(k, candidates.size)]]

            rows.extend([u] * selected.size)
            cols.extend(selected.tolist())

    return _make_graph(rows, cols, n_nodes)

