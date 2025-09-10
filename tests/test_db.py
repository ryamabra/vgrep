import numpy as np
import pytest

from vgrep.db import Db


@pytest.fixture
def db(tmp_path):
    d = Db(tmp_path / "t.db")
    yield d
    d.close()


def test_new_file_needs_encoding(db):
    assert db.upsert_file("/a.jpg", 100.0, 50) is True


def test_unchanged_encoded_file_is_skipped(db):
    db.upsert_file("/a.jpg", 100.0, 50)
    fid = db.conn.execute("SELECT id FROM files").fetchone()["id"]
    db.save_embeddings([(fid, np.ones(4, dtype=np.float32))], 1.0)

    # Same mtime and size, already encoded -> no work.
    assert db.upsert_file("/a.jpg", 100.0, 50) is False


def test_touched_file_is_reencoded(db):
    db.upsert_file("/a.jpg", 100.0, 50)
    fid = db.conn.execute("SELECT id FROM files").fetchone()["id"]
    db.save_embeddings([(fid, np.ones(4, dtype=np.float32))], 1.0)

    # mtime moved -> stale embedding must be cleared and the file requeued.
    assert db.upsert_file("/a.jpg", 200.0, 50) is True
    assert db.count_pending() == 1


def test_interrupted_run_resumes(db):
    for i in range(5):
        db.upsert_file(f"/{i}.jpg", 1.0, 1)
    db.conn.commit()

    ids = [r["id"] for r in db.pending()][:2]
    db.save_embeddings([(i, np.ones(4, dtype=np.float32)) for i in ids], 1.0)

    # Only the un-encoded remainder should come back.
    assert db.count_pending() == 3


def test_model_mismatch_is_refused(db):
    db.check_model_compatibility("model-a", 768)
    with pytest.raises(RuntimeError, match="not comparable"):
        db.check_model_compatibility("model-b", 768)


def test_embeddings_roundtrip(db):
    db.upsert_file("/a.jpg", 1.0, 1)
    fid = db.conn.execute("SELECT id FROM files").fetchone()["id"]
    v = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    db.save_embeddings([(fid, v)], 1.0)

    vecs, ids = db.all_embeddings(4)
    assert ids == [fid]
    np.testing.assert_allclose(vecs[0], v)
