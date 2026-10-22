import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from scipy.sparse import triu
from scipy.sparse.csgraph import connected_components

from graph import GraphData


def grid_coordinates(
    height: int,
    width: int,
) -> np.ndarray:
    """
    Return 2D coordinates associated with image-grid vertices.

    Vertex indexing follows:

        u = i * width + j.
    """

    rows, cols = np.indices(
        (height, width)
    )

    return np.column_stack(
        (
            cols.ravel(),
            -rows.ravel(),
        )
    )

def plot_graph_2d(
    coordinates: np.ndarray,
    graph: GraphData,
    values: np.ndarray | None = None,
    ax=None,
):
    """
    Plot a weighted graph embedded in 2D.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (n_nodes, 2).

    graph : GraphData
        Graph to display.

    values : np.ndarray | None
        Optional scalar value associated with each node.

    ax :
        Optional Matplotlib axis.
    """

    coordinates = np.asarray(
        coordinates,
        dtype=float,
    )

    n_nodes = graph.weights.shape[0]

    if coordinates.shape != (
        n_nodes,
        2,
    ):
        raise ValueError(
            "coordinates must have shape "
            "(n_nodes, 2)."
        )

    if ax is None:
        _, ax = plt.subplots()

    upper = triu(
        graph.weights,
        k=1,
    ).tocoo()

    if upper.nnz > 0:

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

        if values.shape != (n_nodes):
            raise ValueError(
                "values must have shape (n_nodes,)."
            )

        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values,
            s=20,
        )

        plt.colorbar(
            scatter,
            ax=ax,
        )

    ax.set_aspect("equal")

    return ax


def graph_statistics(
    graph: GraphData,
) -> dict:
    """
    Compute simple structural statistics for a graph.
    """

    W = graph.weights

    n_nodes = W.shape[0]

    degrees = np.diff(
        W.indptr
    )

    n_edges = W.nnz // 2

    n_components, _ = connected_components(
        W,
        directed=False,
    )

    stored_weights = W.data

    if stored_weights.size == 0:

        weight_min = np.nan
        weight_max = np.nan
        weight_mean = np.nan

    else:

        weight_min = float(
            stored_weights.min()
        )

        weight_max = float(
            stored_weights.max()
        )

        weight_mean = float(
            stored_weights.mean()
        )

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": float(
            degrees.mean()
        ),
        "min_degree": int(
            degrees.min()
        ),
        "max_degree": int(
            degrees.max()
        ),
        "isolated_nodes": int(
            np.sum(degrees == 0)
        ),
        "n_components": int(
            n_components
        ),
        "weight_min": weight_min,
        "weight_mean": weight_mean,
        "weight_max": weight_max,
    }