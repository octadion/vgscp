"""Unit tests for the CelebA metadata parser (pure, no images/torch needed)."""
import numpy as np

from data.celeba import (load_celeba, parse_attr_file, parse_partition_file,
                         N_SPURIOUS_VALUES)


def _write_fixture(tmp_path):
    # minimal list_attr_celeba.txt with the two attributes we use + a filler
    attr = tmp_path / "list_attr_celeba.txt"
    attr.write_text(
        "6\n"
        "Blond_Hair Male Smiling\n"
        "000001.jpg  1 -1  1\n"   # blond female
        "000002.jpg  1  1 -1\n"   # blond male (rare group)
        "000003.jpg -1  1  1\n"   # non-blond male
        "000004.jpg -1 -1 -1\n"   # non-blond female
        "000005.jpg  1 -1  1\n"
        "000006.jpg -1  1 -1\n",
        encoding="utf-8")
    part = tmp_path / "list_eval_partition.txt"
    part.write_text(
        "000001.jpg 0\n000002.jpg 0\n000003.jpg 0\n"
        "000004.jpg 1\n000005.jpg 2\n000006.jpg 2\n", encoding="utf-8")
    # the loader looks for an image file 000001.jpg to locate img_root
    (tmp_path / "000001.jpg").write_bytes(b"")
    return tmp_path


def test_parse_attr_file(tmp_path):
    _write_fixture(tmp_path)
    files, names, M = parse_attr_file(str(tmp_path / "list_attr_celeba.txt"))
    assert files[0] == "000001.jpg" and len(files) == 6
    assert names == ["Blond_Hair", "Male", "Smiling"]
    # -1 -> 0, 1 -> 1
    assert M[1, names.index("Blond_Hair")] == 1 and M[1, names.index("Male")] == 1   # blond male
    assert M[3, names.index("Blond_Hair")] == 0 and M[3, names.index("Male")] == 0   # non-blond female
    assert set(np.unique(M)) <= {0, 1}


def test_parse_partition_file(tmp_path):
    _write_fixture(tmp_path)
    part = parse_partition_file(str(tmp_path / "list_eval_partition.txt"))
    assert part["000001.jpg"] == 0 and part["000004.jpg"] == 1 and part["000005.jpg"] == 2


def test_load_celeba_groups_and_splits(tmp_path):
    _write_fixture(tmp_path)
    b = load_celeba({"root": str(tmp_path), "n_classes": 2}, seed=0, build_datasets=False)
    # group_id = 2*y + male; blond male (file 2) must be group 3
    assert b.name == "celeba"
    for sp in ("train", "d_learn", "d_cal", "d_test"):
        g = b.group_id[sp]
        y = b.y[sp]
        male = b.spurious_attr[sp]
        assert np.array_equal(g, 2 * y + male)
        assert set(np.unique(g)) <= {0, 1, 2, 3}
    # native train split had 3 files -> bundle train split has exactly those 3
    assert len(b.y["train"]) == 3
    # build_datasets=True must refuse (paths-only loader)
    try:
        load_celeba({"root": str(tmp_path)}, seed=0, build_datasets=True)
        assert False, "should have raised"
    except NotImplementedError:
        pass
