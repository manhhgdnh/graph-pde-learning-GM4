import heapq

import numpy as np

from .data import SeedSet
from .graph import GraphData
from .equations import (
    ClassResult,
    class_potential_columns,
    eikonal_error,
    make_result,
    prepare_eikonal,
    require_boundary_coverage,
    validate_seed_set,
)


def local_eikonal_update(
    neighbor_times: np.ndarray,
    neighbor_weights: np.ndarray,
    potential: float,
    p: float = 2.0,
) -> float:
    times = np.asarray(neighbor_times, dtype=float)
    weights = np.asarray(neighbor_weights, dtype=float)

    if times.ndim != 1 or weights.shape != times.shape:
        raise ValueError("Neighbor times and weights must have the same shape.")

    if np.any(np.isnan(times)) or np.any(times < 0):
        raise ValueError("Neighbor times must be non-negative without NaN.")

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Neighbor weights must be finite and positive.")

    if not np.isfinite(potential) or potential <= 0:
        raise ValueError("potential must be finite and positive.")

    if p not in (1, 2, np.inf):
        raise ValueError("The local solver supports p = 1, 2 or infinity.")

    finite = np.isfinite(times)
    times = times[finite]
    roots = np.sqrt(weights[finite])

    if times.size == 0:
        return np.inf

    if p == np.inf:
        return float(np.min(times + potential / roots))

    order = np.argsort(times, kind="stable")
    times = times[order]
    roots = roots[order]

    # Shift arrival times to reduce cancellation in the quadratic formula.
    origin = times[0]
    shifted = times - origin

    for count in range(1, times.size + 1):
        a = shifted[:count]
        b = roots[:count]

        if p == 1:
            candidate = (
                potential + np.dot(b, a)
            ) / b.sum()

        else:
            q = b ** 2
            A = q.sum()
            B = np.dot(q, a)
            D = np.dot(q, a ** 2)

            discriminant = B ** 2 - A * (D - potential ** 2)
            scale = max(
                B ** 2,
                abs(A * D),
                A * potential ** 2,
                1.0,
            )

            if not np.isfinite(discriminant):
                raise FloatingPointError("The local quadratic solve overflowed.")

            if discriminant < -1e-12 * scale:
                raise FloatingPointError("Invalid local quadratic discriminant.")

            discriminant = max(discriminant, 0.0)
            candidate = (B + np.sqrt(discriminant)) / A

        if not np.isfinite(candidate):
            raise FloatingPointError("The local arrival time overflowed.")

        if count == times.size:
            return float(origin + candidate)

        if candidate <= shifted[count]:
            return float(origin + candidate)

    raise RuntimeError("The local active set was not found.")


def _tentative_time(
    graph: GraphData,
    vertex: int,
    times: np.ndarray,
    accepted: np.ndarray,
    potential: np.ndarray,
    p: float,
) -> float:
    neighbors, weights = graph.neighbors(vertex)
    usable = accepted[neighbors]

    return local_eikonal_update(
        times[neighbors[usable]],
        weights[usable],
        potential[vertex],
        p,
    )


def solve_eikonal_fm(
    graph: GraphData,
    sources,
    potential=1.0,
    p: float = 2.0,
    tolerance: float = 1e-9,
):
    if p not in (1, 2, np.inf):
        raise ValueError("Fast Marching supports p = 1, 2 or infinity.")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    sources, potential, reachable = prepare_eikonal(
        graph,
        sources,
        potential,
    )

    times = np.full(graph.n_nodes, np.inf)
    times[sources] = 0.0

    source_mask = np.zeros(graph.n_nodes, dtype=bool)
    source_mask[sources] = True

    accepted = np.zeros(graph.n_nodes, dtype=bool)
    queue = [(0.0, int(vertex)) for vertex in sources]
    heapq.heapify(queue)

    count = 0

    while queue:
        value, u = heapq.heappop(queue)

        if accepted[u] or value > times[u]:
            continue

        accepted[u] = True
        count += 1

        neighbors, _ = graph.neighbors(u)

        for v in neighbors:
            if accepted[v] or source_mask[v]:
                continue

            candidate = _tentative_time(
                graph,
                int(v),
                times,
                accepted,
                potential,
                p,
            )

            if candidate < times[v]:
                times[v] = candidate
                heapq.heappush(queue, (candidate, int(v)))

    if not np.all(np.isfinite(times[reachable])):
        raise FloatingPointError(
            "Some reachable vertices did not receive finite arrival times."
        )

    residual = eikonal_error(
        graph,
        times,
        sources,
        potential,
        p,
    )

    return make_result(
        times,
        [residual],
        count,
        tolerance,
        reason="completed",
    )


def solve_eikonal_classes(
    graph: GraphData,
    seeds: SeedSet,
    potential=1.0,
    p: float = 2.0,
    tolerance: float = 1e-9,
) -> ClassResult:
    indices = validate_seed_set(graph, seeds)
    require_boundary_coverage(graph, indices)

    classes = np.unique(seeds.labels)
    potentials = class_potential_columns(
        graph,
        potential,
        classes.size,
    )

    fields = np.empty((graph.n_nodes, classes.size))
    results = []

    for column, label in enumerate(classes):
        sources = indices[seeds.labels == label]

        result = solve_eikonal_fm(
            graph,
            sources,
            potential=potentials[:, column],
            p=p,
            tolerance=tolerance,
        )

        fields[:, column] = result.values
        results.append(result)

    if np.any(np.all(~np.isfinite(fields), axis=1)):
        raise ValueError("Some vertices cannot be reached from any class.")

    labels = classes[np.argmin(fields, axis=1)]
    labels[indices] = seeds.labels

    return ClassResult(
        classes=classes,
        fields=fields,
        labels=labels,
        results=results,
    )

