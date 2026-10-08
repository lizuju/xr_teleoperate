from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.check_r1_teleop import check_runtime_version, main


class RuntimeVersionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "repo"
        self.root.mkdir()
        self.paths = {}
        self.files = {}
        for package, module in (("teleimager", "client"), ("televuer", "tv_wrapper")):
            relative = f"teleop/{package}/src/{package}/{module}.py"
            path = self.root / relative
            path.parent.mkdir(parents=True)
            path.write_text(f"MODULE = '{package}'\n", encoding="utf-8")
            self.paths[package] = path
            self.files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                        "commit", "-qm", "test(runtime): fixture"], check=True)
        subprocess.run(["git", "-C", str(self.root), "tag", "production-old"], check=True)

    def write_manifest(self):
        (self.root / ".runtime-release.json").write_text(json.dumps({
            "release": "production-new + runtime-test", "base_commit": "a" * 40, "files": self.files,
        }), encoding="utf-8")

    def output(self):
        buffer = io.StringIO()
        with (redirect_stdout(buffer),
              patch("tools.check_r1_teleop.importlib.util.find_spec",
                    side_effect=lambda package: SimpleNamespace(submodule_search_locations=[str(self.paths[package].parent)])),
              patch("tools.check_r1_teleop.importlib.machinery.PathFinder.find_spec",
                    side_effect=lambda module, locations: SimpleNamespace(origin=str(self.paths[module.split('.')[0]])))):
            check_runtime_version(self.root)
        return buffer.getvalue()

    def test_deployed_snapshot_wins_over_old_dirty_checkout_identity(self):
        self.write_manifest()
        text = self.output()
        self.assertIn("tag=production-old dirty=yes", text)
        self.assertIn("checkout identity only, not the deployed source version", text)
        self.assertIn("production-new + runtime-test base=" + "a" * 40, text)
        self.assertIn("hashes matched 2/2 files", text)
        for path in self.paths.values():
            self.assertIn(str(path), text)
            self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(), text)
        self.assertNotIn("[WARN]", text)

    def test_changed_and_missing_sources_warn_without_claiming_match(self):
        self.write_manifest()
        self.paths["teleimager"].write_text("changed\n", encoding="utf-8")
        self.paths["televuer"].unlink()
        text = self.output()
        self.assertIn("hashes matched 0/2 files", text)
        self.assertIn("Deployed source differs from snapshot", text)
        self.assertIn("teleimager.client is not verified", text)
        self.assertIn("televuer.tv_wrapper version unavailable", text)

    def test_missing_and_invalid_manifest_do_not_abort(self):
        self.assertIn("snapshot unavailable", self.output())
        self.assertIn("checkout HEAD alone cannot identify", self.output())
        manifest = self.root / ".runtime-release.json"
        for contents in ("{", "[]", json.dumps({"release": "r", "base_commit": "a", "files": {"../other": "b" * 64}})):
            manifest.write_text(contents, encoding="utf-8")
            with self.subTest(contents=contents):
                self.assertIn("snapshot unavailable", self.output())

    def test_actual_import_outside_repo_is_reported_even_when_bytes_match(self):
        self.write_manifest()
        outside = Path(self.directory.name) / "other-client.py"
        outside.write_bytes(self.paths["teleimager"].read_bytes())
        self.paths["teleimager"] = outside
        text = self.output()
        self.assertIn("hashes matched 2/2 files", text)
        self.assertIn(str(outside), text)
        self.assertIn("teleimager.client is outside the deployed source directory", text)

    def test_package_discovery_does_not_execute_package_initializers(self):
        for package in self.paths:
            (self.paths[package].parent / "__init__.py").write_text("raise RuntimeError('must not import')\n", encoding="utf-8")
        self.write_manifest()
        buffer = io.StringIO()
        paths = [str(path.parent.parent) for path in self.paths.values()]
        with redirect_stdout(buffer), patch("sys.path", paths):
            check_runtime_version(self.root)
        self.assertIn("Import teleimager.client:", buffer.getvalue())
        self.assertIn("Import televuer.tv_wrapper:", buffer.getvalue())
        self.assertNotIn("must not import", buffer.getvalue())

    def test_warning_only_version_check_preserves_existing_preflight(self):
        with (redirect_stdout(io.StringIO()),
              patch("tools.check_r1_teleop.check_runtime_version") as version,
              patch("tools.check_r1_teleop.check_local_and_https", return_value=object()),
              patch("tools.check_r1_teleop.check_streams", return_value=set()),
              patch("tools.check_r1_teleop.https_probe")):
            self.assertEqual(main(), 0)
        version.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
