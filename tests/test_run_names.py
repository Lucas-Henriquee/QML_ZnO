"""Output names must be readable and preserve existing runs."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from scripts.common.run_names import next_run_name

ROOT = Path(__file__).resolve().parents[1]


class RunNameTests(unittest.TestCase):
    def test_next_name_skips_existing_files_and_directories(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "simulation_01").mkdir()
            (root / "simulation_02").touch()
            self.assertEqual(next_run_name(root, "simulation"), "simulation_03")

    def test_model_names_preserve_previous_outputs(self):
        paths = ["ml/classical_svm_baseline.py", "qml/qsvm_baseline.py", "qml_ibm/qsvm_ibm_hardware.py"]
        for relative in paths:
            with self.subTest(model=relative), TemporaryDirectory() as directory:
                tree = ast.parse((ROOT / "scripts/models" / relative).read_text())
                function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_run_dir")
                namespace = {"Path": Path}
                exec(compile(ast.Module(body=[function], type_ignores=[]), relative, "exec"), namespace)
                args = SimpleNamespace(run_name=None, output_dir=Path(directory), kernel="rbf", scaler="minmax", random_state=42, entanglement="linear", C=10)
                call_args = (args, "md3") if relative.startswith("ml/") else (args, "md3", 6)
                first = namespace["make_run_dir"](*call_args)
                sentinel = first / "preserved.txt"
                sentinel.write_text("original result")
                second = namespace["make_run_dir"](*call_args)
                self.assertNotEqual(first, second)
                self.assertTrue(first.name.endswith("_01"))
                self.assertTrue(second.name.endswith("_02"))
                self.assertEqual(sentinel.read_text(), "original result")
                self.assertEqual((args.C, args.random_state), (10, 42))
                args.run_name = "selected_experiment"
                namespace["make_run_dir"](*call_args)
                with self.assertRaises(FileExistsError):
                    namespace["make_run_dir"](*call_args)


if __name__ == "__main__":
    unittest.main()
