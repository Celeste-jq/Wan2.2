import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class AnalyzeModuleKernelsTests(unittest.TestCase):
    def test_allreduce_is_classified_as_communication(self):
        from tools.analyze_module_kernels import classify_kernel

        self.assertEqual(classify_kernel("hcom_allreduce_612_1316_1"), "communication")
        self.assertEqual(classify_kernel("AllReduce"), "communication")


if __name__ == "__main__":
    unittest.main()
