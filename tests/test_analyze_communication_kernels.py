import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "analyze_communication_kernels.py"


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class AnalyzeCommunicationKernelsTest(unittest.TestCase):
    def test_extracts_communication_summary_from_raw_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "module_kernel_analysis"
            out_dir = tmp / "comm_summary"
            input_dir.mkdir()

            fieldnames = ["kernel_name_norm", "overlap_us", "category"]
            write_csv(
                input_dir / "DIT_LOW_raw_overlap.csv",
                fieldnames,
                [
                    {
                        "kernel_name_norm": "HcomAllGather",
                        "overlap_us": "30",
                        "category": "communication",
                    },
                    {
                        "kernel_name_norm": "HcomAllToAll",
                        "overlap_us": "10",
                        "category": "communication",
                    },
                    {
                        "kernel_name_norm": "FlashAttention",
                        "overlap_us": "60",
                        "category": "attention",
                    },
                ],
            )
            write_csv(
                input_dir / "DIT_HIGH_raw_overlap.csv",
                fieldnames,
                [
                    {
                        "kernel_name_norm": "HcomReduceScatter",
                        "overlap_us": "20",
                        "category": "communication",
                    },
                    {
                        "kernel_name_norm": "MatMulV3",
                        "overlap_us": "80",
                        "category": "matmul",
                    },
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-dir",
                    str(input_dir),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            with (out_dir / "communication_module_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as f:
                module_rows = list(csv.DictReader(f))
            with (out_dir / "communication_kernel_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as f:
                kernel_rows = list(csv.DictReader(f))
            with (out_dir / "communication_type_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as f:
                type_rows = list(csv.DictReader(f))

            self.assertEqual(len(module_rows), 2)
            self.assertEqual(module_rows[0]["module"], "DIT_LOW")
            self.assertEqual(module_rows[0]["communication_total_us"], "40.0")
            self.assertEqual(module_rows[0]["communication_pct"], "40.0")

            self.assertEqual(kernel_rows[0]["module"], "DIT_LOW")
            self.assertEqual(kernel_rows[0]["kernel_name"], "HcomAllGather")
            self.assertEqual(kernel_rows[0]["module_communication_pct"], "75.0")
            self.assertEqual(kernel_rows[0]["module_total_pct"], "30.0")

            alltoall_row = next(row for row in type_rows if row["communication_type"] == "all_to_all")
            self.assertEqual(alltoall_row["module"], "DIT_LOW")
            self.assertEqual(alltoall_row["total_overlap_us"], "10.0")


if __name__ == "__main__":
    unittest.main()
