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


if __name__ == "__main__":
    unittest.main()
