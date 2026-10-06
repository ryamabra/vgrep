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
