from vgrep.scan import iter_images, load_ignore_patterns


def test_ignore_file_skips_indented_comments_and_blank_lines(tmp_path):
    (tmp_path / ".vgrepignore").write_text(
        "\n  # explanation\n*.gif\n", encoding="utf-8"
    )

    assert load_ignore_patterns(tmp_path) == ["*.gif"]


def test_image_discovery_is_deterministic(tmp_path):
    (tmp_path / "z.jpg").write_bytes(b"z")
    (tmp_path / "a.jpg").write_bytes(b"a")

    assert [path.name for path in iter_images(tmp_path)] == ["a.jpg", "z.jpg"]


def test_image_discovery_accepts_uppercase_extensions(tmp_path):
    (tmp_path / "PHOTO.JPEG").write_bytes(b"image")

    assert [path.name for path in iter_images(tmp_path)] == ["PHOTO.JPEG"]


def test_ignore_patterns_match_relative_paths_and_basenames(tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    (album / "keep.jpg").write_bytes(b"image")
    (album / "cover.jpg").write_bytes(b"image")
    (tmp_path / ".vgrepignore").write_text("album/keep.jpg\ncover.jpg\n", encoding="utf-8")

    assert list(iter_images(tmp_path)) == []


def test_hidden_files_and_noise_directories_are_skipped(tmp_path):
    (tmp_path / ".hidden.jpg").write_bytes(b"image")
    cache = tmp_path / "node_modules"
    cache.mkdir()
    (cache / "dependency.png").write_bytes(b"image")
    (tmp_path / "visible.webp").write_bytes(b"image")

    assert [path.name for path in iter_images(tmp_path)] == ["visible.webp"]
