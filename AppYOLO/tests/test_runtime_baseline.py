import sys
import unittest


class RuntimeBaselineTests(unittest.TestCase):
    def test_project_runtime_is_python_313(self) -> None:
        self.assertEqual(
            sys.version_info[:2],
            (3, 13),
            "Run the project test suite with the Python 3.13 launcher or .venv313.",
        )


if __name__ == "__main__":
    unittest.main()
