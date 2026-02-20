import tempfile
import unittest
from pathlib import Path

from main import _cleanup_pycache_dirs


class PycacheCleanupTests(unittest.TestCase):
    def test_cleanup_removes_root_and_nested_pycache_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            root_cache = root / "__pycache__"
            nested_cache = root / "pkg" / "subpkg" / "__pycache__"
            root_cache.mkdir(parents=True)
            nested_cache.mkdir(parents=True)
            (root_cache / "a.pyc").write_bytes(b"x")
            (nested_cache / "b.pyc").write_bytes(b"y")

            removed = _cleanup_pycache_dirs(root)

            self.assertEqual(removed, 2)
            self.assertFalse(root_cache.exists())
            self.assertFalse(nested_cache.exists())


if __name__ == "__main__":
    unittest.main()
