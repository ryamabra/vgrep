import numpy as np
import pytest

from vgrep.index import FlatIndex, search


def test_index_rejects_mismatched_vector_and_id_counts():
    with pytest.raises(ValueError, match="counts must match"):
        FlatIndex(np.zeros((2, 4), dtype=np.float32), [1])


def test_index_rejects_one_dimensional_vectors():
    with pytest.raises(ValueError, match="two-dimensional"):
        FlatIndex(np.zeros(4, dtype=np.float32), [1, 2, 3, 4])


def test_search_requires_positive_result_count():
    index = FlatIndex(np.eye(2, dtype=np.float32), [10, 20])
    with pytest.raises(ValueError, match="positive integer"):
        search(index, np.ones(2, dtype=np.float32), 0)


def test_search_rejects_query_dimension_mismatch():
    index = FlatIndex(np.eye(2, dtype=np.float32), [10, 20])
    with pytest.raises(ValueError, match="query dimension 3"):
        search(index, np.ones(3, dtype=np.float32), 1)
