import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

from .data import SeedSet
from .graph import GraphData
from .equations import (
    ClassResult,
    HybridResult,
    boundary_data,
    class_potential_columns,
    diffusion_terms,
    eikonal_error,
    level_set_terms,
    make_result,
    max_abs,
    prepare_eikonal,
    require_boundary_coverage,
    validate_controls,
    validate_seed_set,
    vertex_array,
)
from .fast_marching import (
    local_eikonal_update,
    solve_eikonal_classes,
)


def _time_step(rates, free, requested=None):
    rate = float(np.max(rates[free])) if np.any(free) else 0.0

    if not np.isfinite(rate) or rate < 0:
        raise FloatingPointError("Invalid time-step bound.")

    if rate == 0:
        return 1.0 if requested is None else requested

    limit = 1.0 / rate

    if requested is None:
        return 0.9 * limit

    if requested > limit * (1.0 + 1e-12):
        raise ValueError(
            "time_step exceeds the explicit bound for the current state."
        )

    return requested


def solve_eikonal_iterative(
    graph: GraphData,
    sources,
    potential=1.0,
    p: float = 2.0,
    method: str = "gauss_seidel",
    max_iter: int = 10000,
    tolerance: float = 1e-9,
):
    validate_controls(max_iter, tolerance)

    if method not in ("jacobi", "gauss_seidel"):
        raise ValueError("method must be jacobi or gauss_seidel.")

    if p not in (1, 2, np.inf):
        raise ValueError("Supported exponents are 1, 2 and infinity.")

    sources, potential, reachable = prepare_eikonal(
        graph,
        sources,
        potential,
    )

    source_mask = np.zeros(graph.n_nodes, dtype=bool)
    source_mask[sources] = True
    vertices = np.flatnonzero(reachable & ~source_mask)

    times = np.full(graph.n_nodes, np.inf)
    times[sources] = 0.0

    history = []
    reason = "max_iter"

    for iteration in range(max_iter + 1):
        error = eikonal_error(
            graph,
            times,
            sources,
            potential,
            p,
        )
        history.append(error)

        if error <= tolerance:
            reason = "steady"
            break

        if iteration == max_iter:
            break

        old = times.copy()
        updated = times.copy()

        order = vertices

        if method == "gauss_seidel" and iteration % 2 == 1:
            order = vertices[::-1]

        for u in order:
            neighbors, weights = graph.neighbors(int(u))

            if method == "jacobi":
                neighbor_times = old[neighbors]
            else:
                neighbor_times = updated[neighbors]

            candidate = local_eikonal_update(
                neighbor_times,
                weights,
                potential[u],
                p,
            )
            updated[u] = min(updated[u], candidate)

        times = updated
        times[sources] = 0.0

    return make_result(
        times,
        history,
        iteration,
        tolerance,
        reason=reason,
    )


def solve_eikonal_classes_iterative(
    graph: GraphData,
    seeds: SeedSet,
    potential=1.0,
    p: float = 2.0,
    method: str = "gauss_seidel",
    max_iter: int = 10000,
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

        result = solve_eikonal_iterative(
            graph,
            sources,
            potential=potentials[:, column],
            p=p,
            method=method,
            max_iter=max_iter,
            tolerance=tolerance,
        )

        fields[:, column] = result.values
        results.append(result)

    if np.any(np.all(~np.isfinite(fields), axis=1)):
        raise RuntimeError(
            "Some vertices remain unreachable. Check coverage or max_iter."
        )

    labels = classes[np.argmin(fields, axis=1)]
    labels[indices] = seeds.labels

    return ClassResult(
        classes=classes,
        fields=fields,
        labels=labels,
        results=results,
    )


def solve_diffusion(
    graph: GraphData,
    initial: np.ndarray,
    model: str = "p_laplacian",
    p: float = 2.0,
    epsilon: float = 1e-6,
    alpha=0.5,
    fidelity: float = 0.0,
    reference=None,
    source=0.0,
    boundary_indices=None,
    boundary_values=0.0,
    time_step: float | None = None,
    final_time: float | None = None,
    max_iter: int = 10000,
    tolerance: float = 1e-8,
):
    validate_controls(max_iter, tolerance, time_step)

    if not np.isfinite(fidelity) or fidelity < 0:
        raise ValueError("fidelity must be finite and non-negative.")

    if final_time is not None:
        if not np.isfinite(final_time) or final_time <= 0:
            raise ValueError("final_time must be finite and positive.")

    values = vertex_array(initial, graph.n_nodes, "initial")

    if reference is None:
        reference = values.copy()
    else:
        reference = vertex_array(reference, graph.n_nodes, "reference")

    source = vertex_array(source, graph.n_nodes, "source")

    fixed, prescribed = boundary_data(
        graph,
        boundary_indices,
        boundary_values,
    )

    free = np.ones(graph.n_nodes, dtype=bool)
    free[fixed] = False
    values[fixed] = prescribed

    history = []
    time = 0.0
    reason = "max_iter"

    for iteration in range(max_iter + 1):
        operator, rates = diffusion_terms(
            graph,
            values,
            model=model,
            p=p,
            epsilon=epsilon,
            alpha=alpha,
        )

        rhs = operator + source - fidelity * (values - reference)
        rates = rates + fidelity

        error = max_abs(rhs[free])
        history.append(error)

        if error <= tolerance:
            reason = "steady"
            break

        if final_time is not None and time >= final_time:
            reason = "final_time"
            break

        if iteration == max_iter:
            break

        step = _time_step(rates, free, time_step)

        if final_time is not None:
            step = min(step, final_time - time)

        values[free] += step * rhs[free]
        values[fixed] = prescribed
        time += step

        if final_time is not None:
            time = min(time, final_time)

        if not np.all(np.isfinite(values)):
            raise FloatingPointError("The diffusion solution overflowed.")

    return make_result(
        values,
        history,
        iteration,
        tolerance,
        time=time,
        reason=reason,
    )


def solve_dirichlet(
    graph: GraphData,
    boundary_indices,
    boundary_values,
    initial=None,
    model: str = "p_laplacian",
    p: float = 2.0,
    epsilon: float = 1e-6,
    alpha=0.5,
    time_step: float | None = None,
    max_iter: int = 10000,
    tolerance: float = 1e-8,
):
    fixed, prescribed = boundary_data(
        graph,
        boundary_indices,
        boundary_values,
    )
    require_boundary_coverage(graph, fixed)

    if initial is None:
        initial = np.full(graph.n_nodes, prescribed.mean())

    return solve_diffusion(
        graph,
        initial,
        model=model,
        p=p,
        epsilon=epsilon,
        alpha=alpha,
        boundary_indices=fixed,
        boundary_values=prescribed,
        time_step=time_step,
        max_iter=max_iter,
        tolerance=tolerance,
    )


def solve_linear_theta(
    graph: GraphData,
    initial: np.ndarray,
    time_step: float,
    n_steps: int,
    theta: float = 1.0,
    fidelity: float = 0.0,
    reference=None,
    source=0.0,
    boundary_indices=None,
    boundary_values=0.0,
    tolerance: float = 1e-8,
):
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be finite and positive.")

    if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer.")

    if not np.isfinite(theta) or not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0, 1].")

    if not np.isfinite(fidelity) or fidelity < 0:
        raise ValueError("fidelity must be finite and non-negative.")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    values = vertex_array(initial, graph.n_nodes, "initial")

    if reference is None:
        reference = values.copy()
    else:
        reference = vertex_array(reference, graph.n_nodes, "reference")

    source = vertex_array(source, graph.n_nodes, "source")

    fixed, prescribed = boundary_data(
        graph,
        boundary_indices,
        boundary_values,
    )
    values[fixed] = prescribed

    free = np.ones(graph.n_nodes, dtype=bool)
    free[fixed] = False

    W = graph.weights
    degrees = np.asarray(W.sum(axis=1)).ravel()

    A = diags(degrees + fidelity, format="csr") - W

    # A conservative spectral bound for theta < 1/2.
    spectral_bound = 2.0 * degrees.max() + fidelity

    if theta < 0.5 and spectral_bound > 0:
        limit = 2.0 / ((1.0 - 2.0 * theta) * spectral_bound)

        if time_step > limit * (1.0 + 1e-12):
            raise ValueError("time_step exceeds the explicit stability bound.")

    forcing_full = source + fidelity * reference
    rhs = forcing_full - A @ values
    history = [max_abs(rhs[free])]

    if not np.any(free):
        return make_result(
            values,
            history,
            0,
            tolerance,
            time=time_step * n_steps,
            reason="final_time",
        )

    A_free = A[free][:, free].tocsc()
    identity = eye(A_free.shape[0], format="csc")

    B = identity + theta * time_step * A_free
    C = identity - (1.0 - theta) * time_step * A_free

    forcing = forcing_full[free].copy()

    if fixed.size:
        forcing -= A[free][:, fixed] @ prescribed

    factorization = None

    if theta > 0:
        factorization = splu(B.tocsc())

    for _ in range(n_steps):
        right_hand_side = (
            C @ values[free]
            + time_step * forcing
        )

        if theta == 0:
            values[free] = right_hand_side
        else:
            values[free] = factorization.solve(right_hand_side)

        values[fixed] = prescribed

        if not np.all(np.isfinite(values)):
            raise FloatingPointError("The theta solution overflowed.")

        rhs = forcing_full - A @ values
        history.append(max_abs(rhs[free]))

    return make_result(
        values,
        history,
        n_steps,
        tolerance,
        time=time_step * n_steps,
        reason="final_time",
    )


def solve_level_set(
    graph: GraphData,
    initial: np.ndarray,
    speed="curvature",
    p: float = 2.0,
    boundary_indices=None,
    boundary_values=0.0,
    time_step: float | None = None,
    final_time: float | None = None,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
):
    validate_controls(max_iter, tolerance, time_step)

    if final_time is not None:
        if not np.isfinite(final_time) or final_time <= 0:
            raise ValueError("final_time must be finite and positive.")

    values = vertex_array(initial, graph.n_nodes, "initial")

    fixed, prescribed = boundary_data(
        graph,
        boundary_indices,
        boundary_values,
    )

    free = np.ones(graph.n_nodes, dtype=bool)
    free[fixed] = False
    values[fixed] = prescribed

    history = []
    time = 0.0
    reason = "max_iter"

    for iteration in range(max_iter + 1):
        rhs, rates = level_set_terms(
            graph,
            values,
            speed=speed,
            p=p,
        )

        error = max_abs(rhs[free])
        history.append(error)

        if error <= tolerance:
            reason = "steady"
            break

        if final_time is not None and time >= final_time:
            reason = "final_time"
            break

        if iteration == max_iter:
            break

        step = _time_step(rates, free, time_step)

        if final_time is not None:
            step = min(step, final_time - time)

        values[free] += step * rhs[free]
        values[fixed] = prescribed
        time += step

        if final_time is not None:
            time = min(time, final_time)

        if not np.all(np.isfinite(values)):
            raise FloatingPointError("The level-set solution overflowed.")

    return make_result(
        values,
        history,
        iteration,
        tolerance,
        time=time,
        reason=reason,
    )


def solve_multiclass_level_set(
    graph: GraphData,
    seeds: SeedSet,
    initial_labels=None,
    p: float = 2.0,
    time_step: float | None = None,
    final_time: float | None = None,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
) -> ClassResult:
    indices = validate_seed_set(graph, seeds)
    require_boundary_coverage(graph, indices)

    classes = np.unique(seeds.labels)

    if initial_labels is not None:
        initial_labels = np.asarray(initial_labels)

        if initial_labels.shape != (graph.n_nodes,):
            raise ValueError("initial_labels must have shape (n_nodes,).")

        if not np.all(np.isin(initial_labels, classes)):
            raise ValueError("initial_labels contains an unknown class.")

    fields = np.empty((graph.n_nodes, classes.size))
    results = []

    for column, label in enumerate(classes):
        prescribed = np.where(seeds.labels == label, 1.0, -1.0)

        if initial_labels is None:
            initial = np.zeros(graph.n_nodes)
        else:
            initial = np.where(initial_labels == label, 1.0, -1.0)

        initial[indices] = prescribed

        result = solve_level_set(
            graph,
            initial,
            speed="curvature",
            p=p,
            boundary_indices=indices,
            boundary_values=prescribed,
            time_step=time_step,
            final_time=final_time,
            max_iter=max_iter,
            tolerance=tolerance,
        )

        fields[:, column] = 0.5 * (result.values + 1.0)
        results.append(result)

    labels = classes[np.argmax(fields, axis=1)]
    labels[indices] = seeds.labels

    return ClassResult(
        classes=classes,
        fields=fields,
        labels=labels,
        results=results,
    )


def solve_hybrid_fm_level_set(
    graph: GraphData,
    seeds: SeedSet,
    potential=1.0,
    eikonal_p: float = 2.0,
    level_set_p: float = 2.0,
    time_step: float | None = None,
    final_time: float | None = None,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
) -> HybridResult:
    front = solve_eikonal_classes(
        graph,
        seeds,
        potential=potential,
        p=eikonal_p,
        tolerance=tolerance,
    )

    if not front.converged:
        raise RuntimeError("Fast Marching did not satisfy the residual tolerance.")

    refinement = solve_multiclass_level_set(
        graph,
        seeds,
        initial_labels=front.labels,
        p=level_set_p,
        time_step=time_step,
        final_time=final_time,
        max_iter=max_iter,
        tolerance=tolerance,
    )

    return HybridResult(
        front=front,
        refinement=refinement,
    )


def solve_poisson_infinity(
    graph: GraphData,
    sources,
    source=1.0,
    alpha=0.4,
    time_step: float | None = None,
    max_iter: int = 10000,
    tolerance: float = 1e-8,
):
    sources, source, _ = prepare_eikonal(
        graph,
        sources,
        source,
    )
    require_boundary_coverage(graph, sources)

    alpha = vertex_array(alpha, graph.n_nodes, "alpha")

    if np.any(alpha < 0) or np.any(alpha > 0.5):
        raise ValueError("This distance-oriented family uses 0 <= alpha <= 0.5.")

    return solve_diffusion(
        graph,
        initial=np.zeros(graph.n_nodes),
        model="biased",
        p=np.inf,
        alpha=alpha,
        source=source,
        boundary_indices=sources,
        boundary_values=0.0,
        time_step=time_step,
        max_iter=max_iter,
        tolerance=tolerance,
    )
