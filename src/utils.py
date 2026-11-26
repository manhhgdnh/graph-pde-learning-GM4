import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from scipy.sparse import triu
from scipy.sparse.csgraph import connected_components

from .graph import GraphData


def grid_coordinates(
    height: int,
    width: int,
) -> np.ndarray:
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive.")

    rows, cols = np.indices((height, width))

    return np.column_stack(
        (cols.ravel(), -rows.ravel())
    )


def plot_graph_2d(
    coordinates: np.ndarray,
    graph: GraphData,
    values: np.ndarray | None = None,
    ax=None,
):
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.shape != (graph.n_nodes, 2):
        raise ValueError("coordinates must have shape (n_nodes, 2).")

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite.")

    if ax is None:
        _, ax = plt.subplots()

    upper = triu(graph.weights, k=1).tocoo()

    if upper.nnz:
        segments = np.stack(
            (
                coordinates[upper.row],
                coordinates[upper.col],
            ),
            axis=1,
        )
        lines = LineCollection(
            segments,
            linewidths=0.5,
            alpha=0.4,
        )
        ax.add_collection(lines)

    if values is None:
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=20,
        )
    else:
        values = np.asarray(values)

        if values.shape != (graph.n_nodes,):
            raise ValueError("values must have shape (n_nodes,).")

        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values,
            s=20,
        )
        plt.colorbar(scatter, ax=ax)

    ax.set_aspect("equal")
    return ax


def graph_statistics(graph: GraphData) -> dict:
    W = graph.weights
    degrees = np.diff(W.indptr)

    n_components, _ = connected_components(
        W,
        directed=False,
    )

    if W.nnz:
        weight_min = float(W.data.min())
        weight_mean = float(W.data.mean())
        weight_max = float(W.data.max())
    else:
        weight_min = np.nan
        weight_mean = np.nan
        weight_max = np.nan

    return {
        "n_nodes": graph.n_nodes,
        "n_edges": graph.n_edges,
        "mean_degree": float(degrees.mean()),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "isolated_nodes": int(np.sum(degrees == 0)),
        "n_components": int(n_components),
        "weight_min": weight_min,
        "weight_mean": weight_mean,
        "weight_max": weight_max,
    }

