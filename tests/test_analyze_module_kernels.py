import unittest
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class AnalyzeModuleKernelsTests(unittest.TestCase):
    def test_allreduce_is_classified_as_communication(self):
        from tools.analyze_module_kernels import classify_kernel

        self.assertEqual(classify_kernel("hcom_allreduce_612_1316_1"), "communication")
        self.assertEqual(classify_kernel("AllReduce"), "communication")

    def test_classify_hardware_class_from_category(self):
        from tools.analyze_module_kernels import classify_hardware_class

        self.assertEqual(classify_hardware_class("attention"), "mixed")
        self.assertEqual(classify_hardware_class("matmul"), "cube")
        self.assertEqual(classify_hardware_class("conv"), "cube")
        self.assertEqual(classify_hardware_class("norm"), "vector")
        self.assertEqual(classify_hardware_class("resize_pad"), "vector")
        self.assertEqual(classify_hardware_class("cast_layout"), "vector")
        self.assertEqual(classify_hardware_class("elementwise"), "vector")
        self.assertEqual(classify_hardware_class("communication"), "communication")
        self.assertEqual(classify_hardware_class("other"), "other")

    def test_load_kernel_rows_assigns_hardware_class(self):
        from tools.analyze_module_kernels import load_kernel_rows

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "kernel_detail.csv"
            csv_path.write_text(
                "\n".join([
                    "kernel_name,start_time,duration",
                    "aclnnAddmm_MatMulV3,0,10",
                    "aclnnCast_CastAiCore_Cast,10,5",
                    "hcom_allGather_1,15,3",
                ]),
                encoding="utf-8",
            )

            _, rows = load_kernel_rows(csv_path)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["hardware_class"], "cube")
        self.assertEqual(rows[1]["hardware_class"], "vector")
        self.assertEqual(rows[2]["hardware_class"], "communication")

    def test_group_hardware_summary(self):
        from tools.analyze_module_kernels import group_hardware_summary

        summary = group_hardware_summary([
            {"hardware_class": "cube", "overlap_us": 10.0},
            {"hardware_class": "cube", "overlap_us": 20.0},
            {"hardware_class": "vector", "overlap_us": 5.0},
        ])

        self.assertEqual(
            summary,
            [
                {
                    "hardware_class": "cube",
                    "call_count": 2,
                    "total_overlap_us": 30.0,
                    "avg_overlap_us": 15.0,
                },
                {
                    "hardware_class": "vector",
                    "call_count": 1,
                    "total_overlap_us": 5.0,
                    "avg_overlap_us": 5.0,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
