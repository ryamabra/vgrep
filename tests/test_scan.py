from vgrep.scan import load_ignore_patterns


def test_ignore_file_skips_indented_comments_and_blank_lines(tmp_path):
    (tmp_path / ".vgrepignore").write_text(
        "\n  # explanation\n*.gif\n", encoding="utf-8"
    )

    assert load_ignore_patterns(tmp_path) == ["*.gif"]
