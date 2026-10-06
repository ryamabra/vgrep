import numpy as np
import pytest

from vgrep.index import FlatIndex


def test_index_rejects_mismatched_vector_and_id_counts():
    with pytest.raises(ValueError, match="counts must match"):
        FlatIndex(np.zeros((2, 4), dtype=np.float32), [1])


def test_index_rejects_one_dimensional_vectors():
    with pytest.raises(ValueError, match="two-dimensional"):
        FlatIndex(np.zeros(4, dtype=np.float32), [1, 2, 3, 4])
