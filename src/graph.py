from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from data import extract_patches

@dataclass
class GraphData:
    """
    Weighted undirected graph represented by a sparse weight matrix.

    The vertex set is V = {0, ..., n_nodes - 1}.

    An edge (u, v) exists if and only if weights[u, v] > 0.

    The matrix is expected to be:
    - square,
    - symmetric,
    - with zero diagonal,
    - with strictly positive stored weights.
    """

    weights: csr_matrix

    def __post_init__(self) -> None:
        if not issparse(self.weights):
            raise TypeError("weights must be a scipy sparse matrix.")

        W = self.weights.tocsr(copy=True)
        n, m = W.shape

        # Merge possible duplicated sparse entries.
        W.sum_duplicates()

        # Remove explicitly stored zeros.
        W.eliminate_zeros()

        # Canonical ordering of column indices.
        W.sort_indices()

        if W.ndim != 2:
            raise ValueError("weights must be a matrix.")

        if n != m:
            raise ValueError("weights must be a square matrix.")

        if not np.all(np.isfinite(W.data)):
            raise ValueError("Graph weights must be finite.")

        if np.any(W.data <= 0):
            raise ValueError(
                "Stored graph weights must be strictly positive."
            )

        if np.any(W.diagonal() != 0):
            raise ValueError(
                "Graph diagonal must be zero."
            )

        self.weights = W
    
    @property
    def n_nodes(self) -> int:
        return self.weights.shape[0]
    
    @property
    def n_edges(self) -> int:
        return self.weights.nnz // 2

    def _check_vertex(self, u : int) -> None:
        if not isinstance(u, (int, np.integer)):
            raise TypeError(
                "Vertex index must be an integer."
        )

        if u < 0 or u >= self.n_nodes:
            raise IndexError(
                f"Vertex {u} is out of range."
        )


    def degree(self, u:int) -> int:
        self._check_vertex(u)

        return (
            self.weights.indptr[u+1] - self.weights.indptr[u]
        )
    
    def weighted_degree(self, u: int) -> float:
        self._check_vertex(u)

        start = self.weights.indptr[u]
        end = self.weights.indptr[u+1]

        return float(
            np.sum(self.weights.data[start:end])
        )

    def neighbors(
        self, 
        u : int,
    ) -> tuple[np.ndarray, np.ndarray]:

        self._check_vertex(u)

        start = self.weights.indptr[u]
        end = self.weights.indptr[u + 1]

        indices = self.weights.indices[start:end]
        weights = self.weights.data[start:end]

        return indices, weights
        

def build_knn_graph(
    X: np.ndarray,
    k: int = 10,
) -> GraphData:
    """
    Build an undirected k-nearest-neighbor graph.

    The directed k-NN relation is symmetrized using the union rule:

        (u, v) is an edge if
        v is among the k nearest neighbors of u
        OR
        u is among the k nearest neighbors of v.

    Constant weights equal to 1 are used in this block.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).

    k : int
        Number of nearest neighbors used for each vertex.

    Returns
    -------
    GraphData
        Undirected sparse k-NN graph.
    """
    
    X = np.asarray(X)
    n_samples = X.shape[0]

    if X.ndim != 2:
        raise ValueError(
            "X must have shape "
            "(n_samples, n_features)."
        )
    
    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )
    

    if k >= n_samples:
        raise ValueError(
            "k must be strictly smaller "
            "than the number of samples."
        )
    
    nn = NearestNeighbors(
        n_neighbors=k+1,
        metric="euclidean",
    )

    nn.fit(X)

    raw_neighbors = nn.kneighbors(
        X,
        return_distance=False,
    )

    neighbors = np.empty(
        (n,samples, k),
        dtype=int,
    )

    for u in range(n_samples):
        candidates = raw_neighbors[u]
        candidates = candidates[candidates != u]
    
        neighbors[u] = candidates[:k]


    rows = np.repeat(
        np.arange(n_samples),
        k,
    )

    cols = neighbors.ravel()

    data = np.ones(
        len(rows),
        dtype=float,
    )

    directed = csr_matrix(
        (data, (rows, cols)),
        shape=(n_samples, n_samples),
    )

    weights = directed.maximum(directed.T)
    weights.setdiag(0)
    weights.eliminate_zeros()
    weights.sort_indices()

    return GraphData(weights=weights)


def build_epsilon_graph(
    X: np.ndarray,
    epsilon: float,
) -> GraphData:
    """
    Build an undirected epsilon-neighborhood graph.

    Two distinct vertices u and v are connected if

        ||X[u] - X[v]||_2 <= epsilon.

    Constant weights equal to 1 are used in this block.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).

    epsilon : float
        Neighborhood radius. Must be strictly positive.

    Returns
    -------
    GraphData
        Undirected sparse epsilon-neighborhood graph.
    """

    X = np.asarray(X)
    n_samples = X.shape[0]

    if X.ndim != 2:
        raise ValueError(
            "X must have shape "
            "(n_samples, n_features)."
        )

    if n_samples == 0:
        raise ValueError(
            "X must contain at least one sample."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be strictly positive."
        )

    nn = NearestNeighbors(
        radius=epsilon,
        metric="euclidean",
    )

    nn.fit(X)

    neighborhoods = nn.radius_neighbors(
        X,
        return_distance=False,
    )

    rows = []
    cols = []

    for u, candidates in enumerate(neighborhoods):

        candidates = candidates[candidates != u]

        rows.extend([u] * len(candidates))

        cols.extend(candidates.tolist())

    if len(rows) == 0:

        weights = csr_matrix(
            (n_samples, n_samples),
            dtype=float,
        )

    else:

        rows = np.asarray(
            rows,
            dtype=int,
        )

        cols = np.asarray(
            cols,
            dtype=int,
        )

        data = np.ones(
            len(rows),
            dtype=float,
        )

        weights = csr_matrix(
            (data, (rows, cols)),
            shape=(
                n_samples,
                n_samples,
            ),
        )

    # The epsilon relation should already be symmetric
    # for a symmetric metric, but we enforce it explicitly
    # to satisfy the GraphData contract.
    
    weights = weights.maximum(weights.T)

    weights.setdiag(0)
    weights.eliminate_zeros()
    weights.sort_indices()

    return GraphData(weights=weights)

def build_grid_graph(
    image: np.ndarray,
    connectivity: int = 4,
) -> GraphData:
    """
    Build a graph from a 2D image grid.

    Each pixel corresponds to one graph vertex.

    Vertex indexing follows row-major order:

        u = i * width + j

    where (i, j) is the pixel coordinate.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W) or
        color image of shape (H, W, C).

    connectivity : {4, 8}
        Pixel neighborhood connectivity.

    Returns
    -------
    GraphData
        Sparse undirected image grid graph with
        constant edge weights equal to 1.
    """

    image = np.asarray(image)
    height, width = image.shape[:2]

    if image.ndim not in (2, 3):
        raise ValueError(
            "image must have shape (H, W) "
            "or (H, W, C)."
        )


    if height == 0 or width == 0:
        raise ValueError(
            "image dimensions must be non-zero."
        )

    if connectivity not in (4, 8):
        raise ValueError(
            "connectivity must be either 4 or 8."
        )

    n_nodes = height * width

    rows = []
    cols = []

    def vertex(i: int, j: int) -> int:
        return i * width + j

    for i in range(height):
        for j in range(width):

            u = vertex(i, j)

            # Right neighbor
            if j + 1 < width:

                v = vertex(i, j + 1)

                rows.extend([u, v])
                cols.extend([v, u])

            # Bottom neighbor
            if i + 1 < height:

                v = vertex(i + 1, j)

                rows.extend([u, v])
                cols.extend([v, u])

            if connectivity == 8:

                # Bottom-right diagonal
                if (
                    i + 1 < height
                    and j + 1 < width
                ):

                    v = vertex(
                        i + 1,
                        j + 1,
                    )

                    rows.extend([u, v])
                    cols.extend([v, u])

                # Bottom-left diagonal
                if (
                    i + 1 < height
                    and j - 1 >= 0
                ):

                    v = vertex(
                        i + 1,
                        j - 1,
                    )

                    rows.extend([u, v])
                    cols.extend([v, u])

    if len(rows) == 0:

        weights = csr_matrix(
            (n_nodes, n_nodes),
            dtype=float,
        )

    else:

        rows = np.asarray(
            rows,
            dtype=int,
        )

        cols = np.asarray(
            cols,
            dtype=int,
        )

        data = np.ones(
            len(rows),
            dtype=float,
        )

        weights = csr_matrix(
            (data, (rows, cols)),
            shape=(
                n_nodes,
                n_nodes,
            ),
        )

    weights.eliminate_zeros()
    weights.sort_indices()

    return GraphData(weights=weights)

def _prepare_features(
    features: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """
    Convert point-cloud or image attributes to a matrix
    of shape (n_nodes, n_features).
    """

    F = np.asarray(
        features,
        dtype=float,
    )

    if F.ndim == 1:
        return F.reshape(
            n_nodes,
            1,
        )

    if F.ndim == 2:

        # Standard point-cloud / feature matrix.
        if F.shape[0] == n_nodes:
            return F

        # Grayscale image.
        if F.size == n_nodes:
            return F.reshape(
                n_nodes,
                1,
            )

    if F.ndim == 3:

        height, width, channels = F.shape

        return F.reshape(
            n_nodes,
            channels,
        )

def _edge_distances(
    features: np.ndarray,
    graph: GraphData,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute distances only for existing graph edges.
    """

    W = graph.weights

    degrees = np.diff(W.indptr)

    rows = np.repeat(
        np.arange(W.shape[0]),
        degrees,
    )

    cols = W.indices

    differences = (
        features[rows]
        - features[cols]
    )

    absolute = np.abs(differences)

    if metric == "euclidean":

        distances = np.sqrt(
            np.sum(
                differences ** 2,
                axis=1,
            )
        )

    elif metric == "manhattan":

        distances = np.sum(
            absolute,
            axis=1,
        )

    elif metric == "chebyshev":

        distances = np.max(
            absolute,
            axis=1,
        )

    else:
        raise ValueError(
            "metric must be 'euclidean', "
            "'manhattan' or 'chebyshev'."
        )

    return rows, cols, distances

def _local_scales(
    features: np.ndarray,
    m: int,
    metric: str,
) -> np.ndarray:
    """
    Compute local scales sigma_u as the distance
    to the M-th nearest neighbor.
    """

    n_nodes = features.shape[0]

    if m < 1 or m >= n_nodes:
        raise ValueError(
            "local_scale_m must satisfy "
            "1 <= M < n_nodes."
        )

    nn = NearestNeighbors(
        n_neighbors=m + 1,
        metric=metric,
    )

    nn.fit(features)

    distances, indices = nn.kneighbors(features)

    sigma = np.empty(
        n_nodes,
        dtype=float,
    )

    for u in range(n_nodes):

        mask = (indices[u] != u)

        local_distances = (distances[u][mask])

        sigma[u] = (local_distances[m - 1])

    if np.any(sigma <= 0):
        raise ValueError(
            "Some local scales are zero. "
            "This can occur with duplicated data points; "
            "choose a larger M or preprocess duplicates."
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
    """
    Compute graph edge weights while preserving the topology.

    Modes
    -----
    constant
        w_uv = 1

    gaussian_global
        w_uv = exp(-d(u,v)^2 / sigma^2)

    gaussian_local
        w_uv = exp(
            -d(u,v)^2 / (sigma_u * sigma_v)
        )

        where sigma_u is the distance from u to its
        M-th nearest neighbor.
    """

    W = graph.weights.copy()

    if mode == "constant":

        W.data[:] = 1.0

        return GraphData(weights=W)

    if features is None:
        raise ValueError(
            "features are required for "
            "Gaussian weights."
        )

    F = _prepare_features(
        features,
        graph.weights.shape[0],
    )

    rows, cols, distances = _edge_distances(
        F,
        graph,
        metric,
    )

    if mode == "gaussian_global":

        if sigma is None:
            raise ValueError(
                "sigma must be provided for "
                "gaussian_global weights."
            )

        new_weights = np.exp(
            -(distances ** 2)
            / (sigma ** 2)
        )

    elif mode == "gaussian_local":

        local_sigma = _local_scales(
            F,
            local_scale_m,
            metric,
        )

        denominator = (
            local_sigma[rows]
            * local_sigma[cols]
        )

        new_weights = np.exp(
            -(distances ** 2)
            / denominator
        )

    else:
        raise ValueError(
            "mode must be 'constant', "
            "'gaussian_global' or "
            "'gaussian_local'."
        )

    # Numerical underflow may produce exact zeros.
    # Mathematically Gaussian similarities are positive.
    new_weights = np.maximum(
        new_weights,
        np.finfo(float).tiny,
    )

    W.data = new_weights

    W.eliminate_zeros()
    W.sort_indices()

    return GraphData(weights=W)

def build_patch_graph(
    image: np.ndarray,
    patch_size: int = 5,
    search_window_size: int = 51,
    k: int = 10,
    patches: np.ndarray | None = None,
) -> GraphData:
    """
    Build a semi-non-local image graph based on patch similarity.

    For each pixel u:
    - define a square spatial search window,
    - compare its patch with the patches inside this window,
    - connect u to the k most similar patches.

    The directed relation is symmetrized using the union rule.

    Constant weights equal to 1 are initially assigned.
    Gaussian patch weights can then be computed with
    compute_weights(...).

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W) or (H, W, C).

    patch_size : int
        Odd width of the patch describing each pixel.

    search_window_size : int
        Odd width of the spatial search window.

    k : int
        Number of most similar patches selected
        inside each search window.

    patches : np.ndarray | None
        Optional precomputed patch matrix.

    Returns
    -------
    GraphData
        Sparse symmetric semi-non-local graph.
    """

    image = np.asarray(image)

    if image.ndim not in (2, 3):
        raise ValueError(
            "image must have shape (H, W) "
            "or (H, W, C)."
        )

    height, width = image.shape[:2]

    n_nodes = height * width

    if (
        search_window_size < 3
        or search_window_size % 2 == 0
    ):
        raise ValueError(
            "search_window_size must be "
            "an odd integer >= 3."
        )

    if k < 1:
        raise ValueError(
            "k must be strictly positive."
        )

    if patches is None:

        patches = extract_patches(
            image,
            patch_size=patch_size,
        )

    else:
        patches = np.asarray(
            patches,
            dtype=float,
        )

        if patches.ndim != 2:
            raise ValueError(
                "patches must be a 2D feature matrix."
            )

        if patches.shape[0] != n_nodes:
            raise ValueError(
                "patches must contain one "
                "descriptor per image pixel."
            )

    radius = search_window_size // 2

    rows = []
    cols = []

    for i in range(height):

        i_min = max(
            0,
            i - radius,
        )

        i_max = min(
            height,
            i + radius + 1,
        )

        for j in range(width):

            j_min = max(
                0,
                j - radius,
            )

            j_max = min(
                width,
                j + radius + 1,
            )

            u = i * width + j

            candidate_rows = np.arange(
                i_min,
                i_max,
            )

            candidate_cols = np.arange(
                j_min,
                j_max,
            )

            grid_i, grid_j = np.meshgrid(
                candidate_rows,
                candidate_cols,
                indexing="ij",
            )

            candidates = (
                grid_i * width
                + grid_j
            ).ravel()

            candidates = candidates[candidates != u]

            if len(candidates) == 0:
                continue

            differences = (
                patches[candidates]
                - patches[u]
            )

            distances_squared = np.sum(
                differences ** 2,
                axis=1,
            )

            n_neighbors = min(
                k,
                len(candidates),
            )

            if (
                n_neighbors
                < len(candidates)
            ):
                selected_positions = np.argpartition(
                    distances_squared,
                    n_neighbors - 1,
                )[:n_neighbors]

                selected = candidates[
                    selected_positions
                ]

            else:
                selected = candidates

            rows.extend(
                [u] * len(selected)
            )

            cols.extend(
                selected.tolist()
            )

    if len(rows) == 0:

        weights = csr_matrix(
            (n_nodes, n_nodes),
            dtype=float,
        )

    else:
        rows = np.asarray(
            rows,
            dtype=int,
        )

        cols = np.asarray(
            cols,
            dtype=int,
        )

        data = np.ones(
            len(rows),
            dtype=float,
        )

        directed = csr_matrix(
            (data, (rows, cols)),
            shape=(
                n_nodes,
                n_nodes,
            ),
        )

        weights = directed.maximum(
            directed.T
        )

        weights.setdiag(0)
        weights.eliminate_zeros()
        weights.sort_indices()

    return GraphData(weights=weights)