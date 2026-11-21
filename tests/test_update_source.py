"""Tests for update source resolution and config overrides."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from echo_app import update_source


class TestUpdateSource(unittest.TestCase):
    def setUp(self):
        self._env_backup = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_default_values(self):
        # Clear any override env vars
        for k in ["ECHO_UPDATE_REPO", "ECHO_UPDATE_RELAY_REPO", "ECHO_UPDATE_RELAY_BASE", "ECHO_UPDATE_LATEST_URL"]:
            os.environ.pop(k, None)

        repo = update_source.get_repo()
        # Default or update_source.json repo
        self.assertIn("echo4d", repo)
        manifest = update_source.get_source_manifest()
        self.assertIsInstance(manifest, dict)
        self.assertIn("repo", manifest)
        self.assertIn("relay_bases", manifest)

    def test_environment_override(self):
        os.environ["ECHO_UPDATE_REPO"] = "CustomOrg/custom-echo"
        os.environ["ECHO_UPDATE_RELAY_BASE"] = "https://cdn.custom.org/updates"
        try:
            self.assertEqual(update_source.get_repo(), "CustomOrg/custom-echo")
            self.assertEqual(update_source.get_relay_bases(), ("https://cdn.custom.org/updates",))
            self.assertEqual(update_source.get_relay_repo(), "CustomOrg/custom-echo-update")
            manifest = update_source.get_source_manifest()
            self.assertTrue(manifest["is_customized"])
            self.assertEqual(manifest["repo"], "CustomOrg/custom-echo")
        finally:
            os.environ.pop("ECHO_UPDATE_REPO", None)
            os.environ.pop("ECHO_UPDATE_RELAY_BASE", None)


if __name__ == "__main__":
    unittest.main()
