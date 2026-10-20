"""Regression checks for accidental additions to the public upload directory."""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_site


class DeploySurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.site = self.root / "site"
        shutil.copytree(check_site.SITE, self.site)

    def check(self) -> int:
        with patch.object(check_site, "SITE", self.site), contextlib.redirect_stdout(io.StringIO()):
            return check_site.main()

    def test_current_public_files_pass(self) -> None:
        self.assertEqual(self.check(), 0)

    def test_nested_backup_is_not_part_of_the_public_surface(self) -> None:
        backup = self.site / "backup"
        backup.mkdir()
        (backup / "internal-notes.md").write_text("private notes", encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "unexpected deploy surface"):
            self.check()

    def test_allowed_filename_cannot_be_a_symlink(self) -> None:
        asset = self.site / "app.js"
        outside = self.root / "outside.js"
        asset.rename(outside)
        asset.symlink_to(outside)
        with self.assertRaisesRegex(AssertionError, "not directories or symlinks"):
            self.check()

    def test_live_answer_cannot_be_replaced_with_retrieval(self) -> None:
        path = self.site / "app.js"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"/demo/answer"', '"/demo/search"'),
            encoding="utf-8",
        )
        with self.assertRaises(AssertionError):
            self.check()

    def test_waking_backend_cannot_be_shown_as_down(self) -> None:
        path = self.site / "app.js"
        path.write_text(
            path.read_text(encoding="utf-8").replace('status("waking"', 'status("down"'),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(AssertionError, "shown as down"):
            self.check()

    def test_refresh_cannot_lose_the_demo_namespace(self) -> None:
        path = self.site / "app.js"
        path.write_text(
            path.read_text(encoding="utf-8").replace("sessionStorage.setItem", "void "),
            encoding="utf-8",
        )
        with self.assertRaises(AssertionError):
            self.check()

    def test_legacy_backend_cannot_enable_the_live_interface(self) -> None:
        path = self.site / "app.js"
        path.write_text(
            path.read_text(encoding="utf-8").replace("health.llm_enabled", "true"), encoding="utf-8"
        )
        with self.assertRaises(AssertionError):
            self.check()


if __name__ == "__main__":
    unittest.main()
