from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import connected_components

from .data import SeedSet
from .graph import GraphData
from .operators import (
    gradient_norm,
    mean_curvature,
    normalized_p_laplacian,
)


@dataclass
class SolverResult:
    values: np.ndarray
    residual_history: np.ndarray
    iterations: int
    converged: bool
    time: float = 0.0
    reason: str = ""

    @property
    def residual(self) -> float:
        return float(self.residual_history[-1])


@dataclass
class ClassResult:
    classes: np.ndarray
    fields: np.ndarray
    labels: np.ndarray
    results: list[SolverResult]

    @property
    def converged(self) -> bool:
        return all(result.converged for result in self.results)


@dataclass
class HybridResult:
    front: ClassResult
    refinement: ClassResult


def max_abs(values: np.ndarray) -> float:
    values = np.asarray(values)

    if values.size == 0:
        return 0.0

    return float(np.max(np.abs(values)))


def make_result(
    values,
    history,
    iterations,
    tolerance,
    time=0.0,
    reason="",
) -> SolverResult:
    history = np.asarray(history, dtype=float)
    residual = float(history[-1])

    return SolverResult(
        values=np.asarray(values, dtype=float).copy(),
        residual_history=history,
        iterations=iterations,
        converged=bool(
            np.isfinite(residual) and residual <= tolerance
        ),
        time=float(time),
        reason=reason,
    )


def vertex_array(
    value,
    n_nodes: int,
    name: str,
    positive: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)

    if array.ndim == 0:
        array = np.full(n_nodes, float(array))
    elif array.shape == (n_nodes,):
        array = array.copy()
    else:
        raise ValueError(f"{name} must be scalar or have shape (n_nodes,).")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")

    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive.")

    return array


def vertex_indices(indices, n_nodes: int) -> np.ndarray:
    indices = np.asarray(indices)

    if indices.ndim != 1:
        raise ValueError("Vertex indices must be one-dimensional.")

    if indices.size == 0:
        return np.empty(0, dtype=int)

    if indices.dtype.kind not in "iu":
        raise TypeError("Vertex indices must be integers.")

    indices = indices.astype(int, copy=True)

    if np.any(indices < 0) or np.any(indices >= n_nodes):
        raise IndexError("A vertex index is out of range.")

    if np.unique(indices).size != indices.size:
        raise ValueError("Vertex indices must be unique.")

    return indices


def validate_seed_set(graph: GraphData, seeds: SeedSet) -> np.ndarray:
    indices = vertex_indices(seeds.indices, graph.n_nodes)

    if indices.size == 0:
        raise ValueError("At least one seed is required.")

    if np.asarray(seeds.labels).shape != indices.shape:
        raise ValueError("Seed labels do not match seed indices.")

    return indices


def boundary_data(graph, indices=None, values=0.0):
    if indices is None:
        return np.empty(0, dtype=int), np.empty(0)

    indices = vertex_indices(indices, graph.n_nodes)
    values = np.asarray(values, dtype=float)

    if values.ndim == 0:
        values = np.full(indices.size, float(values))

    if values.shape != (indices.size,):
        raise ValueError("Boundary values must match boundary indices.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Boundary values must be finite.")

    return indices, values.copy()


def reachable_vertices(graph, sources):
    sources = vertex_indices(sources, graph.n_nodes)

    if sources.size == 0:
        raise ValueError("At least one source is required.")

    _, components = connected_components(
        graph.weights,
        directed=False,
    )

    return np.isin(
        components,
        np.unique(components[sources]),
    )


def require_boundary_coverage(graph, indices):
    reachable = reachable_vertices(graph, indices)

    if not np.all(reachable):
        raise ValueError(
            "Every connected component must contain a prescribed vertex."
        )


def prepare_eikonal(graph, sources, potential):
    sources = vertex_indices(sources, graph.n_nodes)

    if sources.size == 0:
        raise ValueError("At least one source is required.")

    potential = vertex_array(
        potential,
        graph.n_nodes,
        "potential",
        positive=True,
    )
    reachable = reachable_vertices(graph, sources)

    return sources, potential, reachable


def validate_controls(max_iter, tolerance, time_step=None):
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    if time_step is not None:
        if not np.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be finite and positive.")


def constant_potential(graph: GraphData, value: float = 1.0):
    return vertex_array(
        value,
        graph.n_nodes,
        "potential",
        positive=True,
    )


def potential_from_speed(graph: GraphData, speed):
    speed = vertex_array(
        speed,
        graph.n_nodes,
        "speed",
        positive=True,
    )
    return vertex_array(
        1.0 / speed,
        graph.n_nodes,
        "potential",
        positive=True,
    )


def feature_matrix(graph: GraphData, features):
    features = np.asarray(features, dtype=float)

    if features.ndim == 1:
        features = features[:, None]

    if features.ndim != 2 or features.shape[0] != graph.n_nodes:
        raise ValueError("features must have shape (n_nodes, n_features).")

    if features.shape[1] == 0 or not np.all(np.isfinite(features)):
        raise ValueError("features must be non-empty and finite.")

    return features


def gradient_potential(
    graph: GraphData,
    features: np.ndarray,
    floor: float = 1e-3,
):
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be finite and positive.")

    features = feature_matrix(graph, features)
    potential = np.zeros(graph.n_nodes)

    for u in range(graph.n_nodes):
        neighbors, weights = graph.neighbors(u)
        differences = features[neighbors] - features[u]

        potential[u] = np.sqrt(
            np.sum(weights[:, None] * differences ** 2)
        )

    return vertex_array(
        np.maximum(potential, floor),
        graph.n_nodes,
        "potential",
        positive=True,
    )


def seed_distance_potentials(
    graph: GraphData,
    features: np.ndarray,
    seeds: SeedSet,
    floor: float = 1e-3,
):
    validate_seed_set(graph, seeds)
    features = feature_matrix(graph, features)

    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be finite and positive.")

    classes = np.unique(seeds.labels)
    potentials = np.empty((graph.n_nodes, classes.size))

    for column, label in enumerate(classes):
        indices = seeds.indices[seeds.labels == label]
        mean = features[indices].mean(axis=0)

        distances = np.sqrt(
            np.sum((features - mean) ** 2, axis=1)
        )
        potentials[:, column] = np.maximum(distances, floor)

    if not np.all(np.isfinite(potentials)):
        raise FloatingPointError("Class potentials overflowed.")

    return potentials


def class_potential_columns(graph, potential, n_classes):
    potential = np.asarray(potential, dtype=float)

    if potential.ndim <= 1:
        common = vertex_array(
            potential,
            graph.n_nodes,
            "potential",
            positive=True,
        )
        return np.repeat(common[:, None], n_classes, axis=1)

    if potential.shape != (graph.n_nodes, n_classes):
        raise ValueError(
            "Class potentials must have shape (n_nodes, n_classes)."
        )

    if not np.all(np.isfinite(potential)) or np.any(potential <= 0):
        raise ValueError("Class potentials must be finite and positive.")

    return potential.copy()


def eikonal_residual(
    graph: GraphData,
    times: np.ndarray,
    sources,
    potential=1.0,
    p: float = 2.0,
):
    sources, potential, reachable = prepare_eikonal(
        graph,
        sources,
        potential,
    )

    times = np.asarray(times, dtype=float)

    if times.shape != (graph.n_nodes,):
        raise ValueError("times must have shape (n_nodes,).")

    if np.any(np.isnan(times)) or np.any(times < 0):
        raise ValueError("times must be non-negative and must not contain NaN.")

    source_mask = np.zeros(graph.n_nodes, dtype=bool)
    source_mask[sources] = True
    free = reachable & ~source_mask

    residual = np.full(graph.n_nodes, np.nan)
    residual[sources] = times[sources]

    if not np.all(np.isfinite(times[reachable])):
        residual[free] = np.inf
        return residual

    safe_times = times.copy()
    safe_times[~reachable] = 0.0

    norms = gradient_norm(
        safe_times,
        graph,
        p=p,
        direction="minus",
    )
    residual[free] = norms[free] - potential[free]

    return residual


def eikonal_error(graph, times, sources, potential=1.0, p=2.0):
    residual = eikonal_residual(
        graph,
        times,
        sources,
        potential,
        p,
    )
    return max_abs(residual[~np.isnan(residual)])


def gradient_bounds(graph: GraphData, p: float):
    if p != np.inf:
        if not np.isfinite(p) or p < 1:
            raise ValueError("p must satisfy p >= 1 or p = infinity.")

    bounds = np.zeros(graph.n_nodes)

    for u in range(graph.n_nodes):
        _, weights = graph.neighbors(u)

        if weights.size == 0:
            continue

        roots = np.sqrt(weights)

        if p == np.inf:
            bounds[u] = roots.max()
        else:
            bounds[u] = np.sum(roots ** p) ** (1.0 / p)

    return bounds


def diffusion_terms(
    graph: GraphData,
    values: np.ndarray,
    model: str = "p_laplacian",
    p: float = 2.0,
    epsilon: float = 1e-6,
    alpha=0.5,
):
    values = vertex_array(values, graph.n_nodes, "values")

    if model == "normalized_p":
        rhs = normalized_p_laplacian(values, graph, p)
        return rhs, np.ones(graph.n_nodes)

    if model == "infinity":
        plus = gradient_norm(values, graph, np.inf, "plus")
        minus = gradient_norm(values, graph, np.inf, "minus")

        return 0.5 * (plus - minus), gradient_bounds(graph, np.inf)

    if model not in ("p_laplacian", "biased"):
        raise ValueError("Unknown diffusion model.")

    if model == "biased":
        alpha = vertex_array(alpha, graph.n_nodes, "alpha")

        if np.any(alpha < 0) or np.any(alpha > 1):
            raise ValueError("alpha must lie in [0, 1].")

        if p == np.inf:
            plus = gradient_norm(values, graph, np.inf, "plus")
            minus = gradient_norm(values, graph, np.inf, "minus")

            rhs = alpha * plus - (1.0 - alpha) * minus
            return rhs, gradient_bounds(graph, np.inf)

        if not np.isfinite(p) or p < 2:
            raise ValueError("The finite biased model requires p >= 2.")

    else:
        if not np.isfinite(p) or p < 1:
            raise ValueError("The p-Laplacian model requires finite p >= 1.")

        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and non-negative.")

        if p < 2 and epsilon == 0:
            raise ValueError("Use epsilon > 0 when p < 2.")

    W = graph.weights

    rows = np.repeat(
        np.arange(graph.n_nodes),
        np.diff(W.indptr),
    )
    delta = values[W.indices] - values[rows]

    if model == "p_laplacian":
        coefficients = (
            W.data ** (p / 2.0)
            * (delta ** 2 + epsilon ** 2) ** ((p - 2.0) / 2.0)
        )

    else:
        factors = alpha[rows].copy()

        negative = delta < 0
        factors[negative] = 1.0 - alpha[rows[negative]]

        zero = delta == 0
        factors[zero] = np.maximum(
            alpha[rows[zero]],
            1.0 - alpha[rows[zero]],
        )

        coefficients = (
            factors
            * W.data ** ((p - 1.0) / 2.0)
            * np.abs(delta) ** (p - 2.0)
        )

    rhs = np.bincount(
        rows,
        weights=coefficients * delta,
        minlength=graph.n_nodes,
    )
    rates = np.bincount(
        rows,
        weights=coefficients,
        minlength=graph.n_nodes,
    )
    rates *= max(1.0, p - 1.0)

    if not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(rates)):
        raise FloatingPointError("Diffusion terms overflowed.")

    return rhs, rates


def level_set_terms(
    graph: GraphData,
    values: np.ndarray,
    speed="curvature",
    p: float = 2.0,
):
    values = vertex_array(values, graph.n_nodes, "values")

    if isinstance(speed, str):
        if speed != "curvature":
            raise ValueError("Unknown speed model.")

        velocity = mean_curvature(values, graph)
    else:
        velocity = vertex_array(speed, graph.n_nodes, "speed")

    plus = gradient_norm(values, graph, p, "plus")
    minus = gradient_norm(values, graph, p, "minus")

    rhs = (
        np.maximum(velocity, 0.0) * plus
        - np.maximum(-velocity, 0.0) * minus
    )
    rates = np.abs(velocity) * gradient_bounds(graph, p)

    return rhs, rates
