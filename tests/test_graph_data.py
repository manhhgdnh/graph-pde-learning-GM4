import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.graph import GraphData


@pytest.fixture
def graph():
    W = csr_matrix([
        [0, 2.0, 1.0, 0],
        [2.0, 0, 3.0, 0],
        [1.0, 3.0, 0, 4.0],
        [0, 0, 4.0, 0],
    ])

    return GraphData(W)


def test_n_nodes(graph):
    assert graph.n_nodes == 4


def test_n_edges(graph):
    assert graph.n_edges == 4


def test_degree(graph):
    assert graph.degree(0) == 2
    assert graph.degree(1) == 2
    assert graph.degree(2) == 3
    assert graph.degree(3) == 1


def test_weighted_degree(graph):
    assert graph.weighted_degree(0) == pytest.approx(3.0)
    assert graph.weighted_degree(1) == pytest.approx(5.0)
    assert graph.weighted_degree(2) == pytest.approx(8.0)
    assert graph.weighted_degree(3) == pytest.approx(4.0)


def test_negative_vertex(graph):
    with pytest.raises(IndexError):
        graph.degree(-1)


def test_vertex_out_of_range(graph):
    with pytest.raises(IndexError):
        graph.degree(4)


def test_invalid_vertex_type(graph):
    with pytest.raises(TypeError):
        graph.degree(1.5)


def test_non_square_matrix():
    W = csr_matrix([
        [0, 1, 2],
        [1, 0, 3],
    ])

    with pytest.raises(ValueError):
        GraphData(W)


def test_nonzero_diagonal():
    W = csr_matrix([
        [1, 2],
        [2, 0],
    ])

    with pytest.raises(ValueError):
        GraphData(W)


def test_negative_weight():
    W = csr_matrix([
        [0, -2],
        [-2, 0],
    ])

    with pytest.raises(ValueError):
        GraphData(W)