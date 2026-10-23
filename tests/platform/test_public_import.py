"""These tests run without PyTorch or UP installed."""
import subprocess
import sys
import unittest


class PublicImportTest(unittest.TestCase):
    def test_import_does_not_load_torch_up_or_vendor(self):
        command = "import up_platform,sys; assert 'torch' not in sys.modules; assert 'up' not in sys.modules; assert not any(n.startswith('up_platform.backends.') for n in sys.modules)"
        subprocess.run([sys.executable, "-c", command], check=True)

    def test_selection_requires_an_explicit_purpose(self):
        from up_platform import configure_platform, get_platform
        with self.assertRaises(ValueError):
            configure_platform({}, purpose="eval")
        with self.assertRaises(RuntimeError):
            get_platform()


if __name__ == "__main__":
    unittest.main()
