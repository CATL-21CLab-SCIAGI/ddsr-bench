from pathlib import Path

import h5py
import numpy as np

from ddsr_bench.benchmarks.scicode.loader import load_targets


def _file(path: Path) -> None:
    with h5py.File(path, "w") as file:
        step = file.create_group("1.1")
        step.create_group("test1").create_dataset("value", data=3)
        step.create_group("test2").create_dataset("value", data=np.bytes_("text"))

        pair = step.create_group("test3")
        pair.create_dataset("var1", data=1)
        pair.create_dataset("var2", data=np.bytes_("two"))

        sequence = step.create_group("test4").create_group("value")
        items = sequence.create_group("list")
        items.create_dataset("var1", data=4)
        items.create_dataset("var2", data=5)

        mapping = step.create_group("test5").create_group("value")
        mapping.create_dataset("2.5", data=6)
        mapping.create_dataset("name", data=np.bytes_("seven"))

        sparse = step.create_group("test6").create_group("value")
        matrix = sparse.create_group("sparse_matrix")
        matrix.create_dataset("data", data=[8, 9])
        matrix.create_dataset("indices", data=[0, 1])
        matrix.create_dataset("indptr", data=[0, 1, 2])
        matrix.create_dataset("shape", data=[2, 2])


def test_published_hdf5_layout(tmp_path: Path) -> None:
    path = tmp_path / "targets.h5"
    _file(path)

    targets = load_targets("1.1", 6, path)

    assert targets[:5] == [
        3,
        "text",
        (1, "two"),
        [4, 5],
        {2.5: 6, "name": "seven"},
    ]
    np.testing.assert_array_equal(targets[5].toarray(), [[8, 0], [0, 9]])
