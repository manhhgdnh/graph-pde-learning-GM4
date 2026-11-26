import numpy as np
import pytest

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.data import SeedSet, extract_patches, load_image
from src.graph import (
    GraphData,
    build_grid_graph,
    build_knn_graph,
    build_patch_graph,
    compute_weights,
)
from src.operators import (
    gradient_norm,
    normalized_p_laplacian,
)
from src.equations import (
    eikonal_error,
    gradient_potential,
    seed_distance_potentials,
)
from src.fast_marching import (
    local_eikonal_update,
    solve_eikonal_fm,
    solve_eikonal_classes,
)
from src.iterative import (
    solve_eikonal_iterative,
    solve_diffusion,
    solve_dirichlet,
    solve_linear_theta,
    solve_level_set,
    solve_hybrid_fm_level_set,
    solve_poisson_infinity,
)


def path_graph(n_nodes, weight=1.0):
    W = np.zeros((n_nodes, n_nodes))

    for u in range(n_nodes - 1):
        W[u, u + 1] = weight
        W[u + 1, u] = weight

    return GraphData(csr_matrix(W))


@pytest.mark.parametrize(
    "p, expected",
    [(1, 0.5), (2, 1.0 / np.sqrt(2.0)), (np.inf, 1.0)],
)
def test_local_two_equal_neighbors(p, expected):
    value = local_eikonal_update(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        potential=1.0,
        p=p,
    )
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_local_weight_convention(p):
    value = local_eikonal_update(
        np.array([0.0]),
        np.array([0.25]),
        potential=1.0,
        p=p,
    )
    assert value == pytest.approx(2.0)


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_local_ignores_late_neighbor(p):
    value = local_eikonal_update(
        np.array([0.0, 100.0]),
        np.array([1.0, 1.0]),
        potential=1.0,
        p=p,
    )
    assert value == pytest.approx(1.0)


def test_gradient_minus_convention():
    graph = path_graph(2, weight=0.25)

    actual = gradient_norm(
        np.array([0.0, 2.0]),
        graph,
        p=np.inf,
        direction="minus",
    )
    np.testing.assert_allclose(actual, [0.0, 1.0])


def test_graph_rejects_asymmetry():
    with pytest.raises(ValueError):
        GraphData(csr_matrix([[0.0, 1.0], [0.0, 0.0]]))


def test_knn_builder_and_gaussian_weights():
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])

    graph = build_knn_graph(X, k=1)
    weighted = compute_weights(
        graph,
        X,
        mode="gaussian_global",
        sigma=1.0,
    )

    assert graph.n_nodes == 4
    assert (weighted.weights - weighted.weights.T).nnz == 0
    assert np.all(weighted.weights.data > 0)
    assert np.all(weighted.weights.data <= 1)


def test_image_and_patch_interfaces():
    image = np.arange(9, dtype=float).reshape(3, 3)

    patches = extract_patches(image, patch_size=3)
    graph = build_patch_graph(
        image,
        patch_size=3,
        search_window_size=3,
        k=2,
        patches=patches,
    )

    assert patches.shape == (9, 9)
    assert graph.n_nodes == 9


def test_load_image(tmp_path):
    from PIL import Image

    path = tmp_path / "small.png"
    Image.fromarray(
        np.array([[0, 255]], dtype=np.uint8)
    ).save(path)

    image = load_image(path, grayscale=True)
    np.testing.assert_allclose(image, [[0.0, 1.0]])


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_path_arrival_times(p):
    graph = path_graph(6)

    result = solve_eikonal_fm(graph, [0], p=p)

    np.testing.assert_allclose(result.values, np.arange(6))
    assert result.converged
    assert result.values[0] == 0.0


@pytest.mark.parametrize("p", [1, 2, np.inf])
@pytest.mark.parametrize("method", ["jacobi", "gauss_seidel"])
def test_fm_matches_iterative(p, method):
    graph = GraphData(csr_matrix([
        [0.0, 1.0, 0.4, 0.0, 0.0],
        [1.0, 0.0, 0.8, 0.5, 0.0],
        [0.4, 0.8, 0.0, 1.0, 0.3],
        [0.0, 0.5, 1.0, 0.0, 0.7],
        [0.0, 0.0, 0.3, 0.7, 0.0],
    ]))
    potential = np.array([1.0, 0.8, 1.2, 0.9, 1.1])

    fm = solve_eikonal_fm(
        graph,
        [0],
        potential=potential,
        p=p,
    )
    iterative = solve_eikonal_iterative(
        graph,
        [0],
        potential=potential,
        p=p,
        method=method,
        tolerance=1e-10,
    )

    assert fm.converged
    assert iterative.converged

    np.testing.assert_allclose(
        iterative.values,
        fm.values,
        rtol=1e-8,
        atol=1e-8,
    )

    assert eikonal_error(
        graph,
        iterative.values,
        [0],
        potential,
        p,
    ) < 1e-8


def test_infinity_fm_matches_dijkstra():
    graph = GraphData(csr_matrix([
        [0.0, 1.0, 0.25, 0.0],
        [1.0, 0.0, 0.5, 0.2],
        [0.25, 0.5, 0.0, 1.0],
        [0.0, 0.2, 1.0, 0.0],
    ]))

    lengths = graph.weights.copy()
    lengths.data = 1.0 / np.sqrt(lengths.data)

    expected = dijkstra(
        lengths,
        directed=False,
        indices=0,
    )
    actual = solve_eikonal_fm(graph, [0], p=np.inf)

    np.testing.assert_allclose(actual.values, expected)


def test_unreachable_vertices_remain_infinite():
    graph = GraphData(csr_matrix([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]))

    result = solve_eikonal_fm(graph, [0])

    np.testing.assert_allclose(result.values[:2], [0.0, 1.0])
    assert np.all(np.isinf(result.values[2:]))
    assert result.converged


def test_multiclass_is_not_merged_sources():
    graph = path_graph(3)
    seeds = SeedSet(
        np.array([0, 2]),
        np.array([4, 9]),
    )

    separate = solve_eikonal_classes(graph, seeds, p=2)
    merged = solve_eikonal_fm(graph, [0, 2], p=2)

    np.testing.assert_allclose(separate.fields[1], [1.0, 1.0])
    assert merged.values[1] == pytest.approx(1.0 / np.sqrt(2.0))

    np.testing.assert_array_equal(
        separate.labels[seeds.indices],
        seeds.labels,
    )


def test_dirichlet_linear_interpolation():
    graph = path_graph(5)

    result = solve_dirichlet(
        graph,
        boundary_indices=[0, 4],
        boundary_values=[0.0, 1.0],
        tolerance=1e-10,
    )

    assert result.converged
    np.testing.assert_allclose(
        result.values,
        np.linspace(0.0, 1.0, 5),
        atol=1e-8,
    )


def test_dirichlet_requires_boundary_coverage():
    graph = GraphData(csr_matrix(np.zeros((2, 2))))

    with pytest.raises(ValueError):
        solve_dirichlet(graph, [0], [0.0])


def test_explicit_rejects_large_step():
    graph = path_graph(3)

    with pytest.raises(ValueError):
        solve_diffusion(
            graph,
            initial=np.array([0.0, 1.0, 0.0]),
            time_step=10.0,
            max_iter=2,
        )


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_linear_theta_against_exact_two_vertex_solution(theta):
    graph = path_graph(2)
    final_time = 1.0

    result = solve_linear_theta(
        graph,
        initial=np.array([0.0, 1.0]),
        time_step=0.01,
        n_steps=100,
        theta=theta,
    )

    amplitude = 0.5 * np.exp(-2.0 * final_time)
    expected = np.array([0.5 - amplitude, 0.5 + amplitude])

    np.testing.assert_allclose(
        result.values,
        expected,
        atol=2e-3,
    )
    assert result.time == pytest.approx(final_time)


def test_normalized_p_two_is_weighted_mean():
    graph = path_graph(3)
    values = np.array([0.0, 1.0, 0.0])

    actual = normalized_p_laplacian(values, graph, p=2)

    np.testing.assert_allclose(actual, [1.0, -1.0, 1.0])


def test_level_set_preserves_constant_data():
    graph = path_graph(4)
    initial = np.full(4, 0.3)

    result = solve_level_set(graph, initial)

    np.testing.assert_allclose(result.values, initial)
    assert result.converged


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_level_set_keeps_bounds_and_boundary(p):
    graph = path_graph(5)

    result = solve_level_set(
        graph,
        initial=np.array([-1.0, -0.5, 0.2, 0.6, 1.0]),
        speed=np.array([1.0, -0.5, 0.7, -1.0, 1.0]),
        p=p,
        boundary_indices=[0, 4],
        boundary_values=[-1.0, 1.0],
        max_iter=20,
    )

    assert result.values.min() >= -1.0 - 1e-12
    assert result.values.max() <= 1.0 + 1e-12
    np.testing.assert_allclose(result.values[[0, 4]], [-1.0, 1.0])


def test_poisson_zero_bias_matches_infinity_eikonal():
    graph = path_graph(5)

    fm = solve_eikonal_fm(graph, [0], p=np.inf)
    poisson = solve_poisson_infinity(
        graph,
        [0],
        alpha=0.0,
        tolerance=1e-10,
    )

    assert poisson.converged

    np.testing.assert_allclose(
        poisson.values,
        fm.values,
        atol=1e-8,
    )


def test_potentials_are_positive():
    graph = path_graph(3)
    features = np.array([[0.0], [1.0], [2.0]])
    seeds = SeedSet(np.array([0, 2]), np.array([0, 1]))

    gradient = gradient_potential(graph, features)
    distances = seed_distance_potentials(graph, features, seeds)

    assert gradient.shape == (3,)
    assert distances.shape == (3, 2)
    assert np.all(gradient > 0)
    assert np.all(distances > 0)


def test_hybrid_preserves_seed_labels():
    image = np.zeros((3, 3))
    graph = build_grid_graph(image, connectivity=4)
    seeds = SeedSet(np.array([0, 8]), np.array([0, 1]))

    result = solve_hybrid_fm_level_set(
        graph,
        seeds,
        max_iter=20,
    )

    np.testing.assert_array_equal(
        result.front.labels[seeds.indices],
        seeds.labels,
    )
    np.testing.assert_array_equal(
        result.refinement.labels[seeds.indices],
        seeds.labels,
    )

