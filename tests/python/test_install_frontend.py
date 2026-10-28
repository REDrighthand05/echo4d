"""Exercise frontend provisioning in real, isolated Git working trees."""
from pathlib import Path
import subprocess
import sys

ASSETS = ("quality.html", "quality.css", "quality.mjs", "quality-intake.mjs")
SCRIPT = Path(__file__).resolve().parents[2] / "tools/install_frontend.py"


def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    git("init")
    (root / "dist").mkdir()
    for name in ASSETS:
        (root / "dist" / name).write_text("canonical", encoding="utf-8")
    git("add", "dist")
    git("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-m", "canonical assets")
    return root


def install(root, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "--source", "HEAD", *args],
                          cwd=root, capture_output=True, text=True)


def test_missing_assets_install_from_git(tmp_path):
    root = repository(tmp_path)
    for name in ASSETS:
        (root / "dist" / name).unlink()
    result = install(root)
    assert result.returncode == 0, result.stderr
    assert all((root / "dist" / name).read_text() == "canonical" for name in ASSETS)


def test_directory_symlink_cannot_write_outside_worktree_even_with_force(tmp_path):
    root = repository(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    for name in ASSETS:
        (root / "dist" / name).unlink()
    (root / "dist").rmdir()
    (root / "dist").symlink_to(outside, target_is_directory=True)
    for args in ((), ("--force",), ("--check",)):
        result = install(root, *args)
        assert result.returncode == 1, result.stdout
        assert "BLOCKED unsafe target" in result.stderr
        assert list(outside.iterdir()) == []


def test_force_preserves_uncommitted_edits_and_leaf_symlink(tmp_path):
    root = repository(tmp_path)
    edited = root / "dist" / "quality.html"
    edited.write_text("local edits", encoding="utf-8")
    outside = tmp_path / "outside.css"
    outside.write_text("external sentinel", encoding="utf-8")
    linked = root / "dist" / "quality.css"
    linked.unlink()
    linked.symlink_to(outside)
    result = install(root, "--force")
    assert result.returncode == 1
    assert "BLOCKED uncommitted local edits" in result.stderr
    assert "BLOCKED unsafe target" in result.stderr
    assert edited.read_text() == "local edits"
    assert outside.read_text() == "external sentinel"
