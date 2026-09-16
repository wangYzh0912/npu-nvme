import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from user_environment import isolated_environment, read_profile, verify_library_paths, with_python_identity
from experiments.benchmarks.environment_upgrade_inventory import file_record, path_info
from experiments.benchmarks.model_compatibility import probe


class UserEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.toolkit = self.root / "stack/cann/aarch64-linux"
        for directory in (self.toolkit / "lib64", self.toolkit.parent / "opp",
                          self.root / "stack/env/bin", self.root / "repo/python"):
            directory.mkdir(parents=True)
        self.python = self.root / "stack/env/bin/python"
        self.script = self.toolkit / "setenv.bash"
        self.library = self.toolkit / "lib64/libnpu_nvme.so"
        self.driver = self.root / "driver.info"
        self.python.touch()
        # If the launcher accidentally sources this, the test must fail.
        self.script.write_text("exit 99\n")
        self.library.write_bytes(b"library")
        self.driver.write_text("Version=24.1.rc3\n")
        self.manifest_path = self.root / "environments.json"
        self.manifest = {
            "schema_version": 1, "default": "old", "private_root": str(self.root / "stack"),
            "driver_version_file": str(self.driver), "expected_driver_version": "24.1.rc3",
            "profiles": {"old": {"python": str(self.python), "toolkit": str(self.toolkit),
                                   "set_env": str(self.script), "library": str(self.library)},
                         "candidate": None}}
        self.save()

    def save(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def profile(self, name="old"):
        return read_profile(self.manifest_path, name, self.root / "repo")

    def test_inherited_python_cann_and_shell_hooks_are_removed(self):
        incoming = {"HOME": "/unchanged", "PATH": "/other/bin", "PYTHONPATH": "/other/python",
                    "PYTHONHOME": "/other/python", "LD_LIBRARY_PATH": "/other/lib",
                    "LD_PRELOAD": "/bad.so", "CONDA_PREFIX": "/other", "BASH_ENV": "/hook",
                    "ASCEND_CUSTOM_OPP_PATH": "/other/opp", "DEVICE_ID": "3"}
        env = isolated_environment(self.profile(), self.root / "repo", incoming)
        self.assertEqual(env["HOME"], "/unchanged")
        self.assertEqual(env["DEVICE_ID"], "3")
        self.assertEqual(env["ASCEND_OPP_PATH"], str(self.toolkit.parent / "opp"))
        for key in ("PYTHONHOME", "LD_PRELOAD", "CONDA_PREFIX", "BASH_ENV", "ASCEND_CUSTOM_OPP_PATH"):
            self.assertNotIn(key, env)
        for key in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH"):
            self.assertNotIn("/other", env[key])
            self.assertNotIn("", env[key].split(os.pathsep))
        self.assertEqual(incoming["PATH"], "/other/bin")

    def test_candidate_is_not_implicitly_available_or_promoted(self):
        with self.assertRaisesRegex(ValueError, "no verified installation"):
            self.profile("candidate")
        self.manifest["default"] = "candidate"
        self.save()
        with self.assertRaisesRegex(ValueError, "promotion"):
            self.profile("default")

    def test_driver_change_requires_new_audit(self):
        self.driver.write_text("Version=25.5.0\n")
        with self.assertRaisesRegex(ValueError, "driver version changed"):
            self.profile()

    def test_candidate_symlink_cannot_escape_private_installation(self):
        outside = self.root / "outside-python"
        outside.touch()
        self.python.unlink()
        self.python.symlink_to(outside)
        self.manifest["profiles"]["candidate"] = self.manifest["profiles"]["old"].copy()
        self.save()
        with self.assertRaisesRegex(ValueError, "outside private_root"):
            self.profile("candidate")

    def test_library_change_changes_environment_identity(self):
        first = self.profile()["environment_id"]
        self.library.write_bytes(b"rebuilt")
        self.assertNotEqual(first, self.profile()["environment_id"])

    def test_python_package_change_invalidates_environment_identity(self):
        profile = self.profile()
        responses = [subprocess.CompletedProcess([], 0, json.dumps({"python": "3.9.25", "packages": packages}), "")
                     for packages in ([['mindspore', '2.5.0']], [['mindspore', '2.7.2']])]
        with patch("user_environment.subprocess.run", side_effect=responses):
            self.assertNotEqual(with_python_identity(profile)["environment_id"],
                                with_python_identity(profile)["environment_id"])

    def test_missing_dependencies_and_wrong_cann_are_rejected(self):
        for output in ("libascendcl.so => not found\n",
                       "libascendcl.so => /other/cann/lib/libascendcl.so (0x123)\n",
                       "libascendcl.so => /other/stubs/libascendcl.so (0x123)\n"):
            with self.subTest(output=output), patch("user_environment.subprocess.run", return_value=
                    subprocess.CompletedProcess([], 0, output, "")):
                with self.assertRaises(ValueError):
                    verify_library_paths(self.profile(), {})

    def test_matching_cann_is_accepted(self):
        path = self.toolkit / "lib64/libascendcl.so"
        with patch("user_environment.subprocess.run", return_value=subprocess.CompletedProcess(
                [], 0, f"libascendcl.so => {path} (0x123)\n", "")):
            self.assertEqual(verify_library_paths(self.profile(), {})["libascendcl.so"], str(path))


class InventoryAndModelTests(unittest.TestCase):
    def test_permission_error_is_not_reported_as_missing(self):
        with patch.object(Path, "stat", side_effect=PermissionError("denied")):
            result = path_info("/restricted")
        self.assertIsNone(result["exists"])
        self.assertIn("PermissionError", result["stat_error"])
        with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            self.assertIn("read_error", file_record("/restricted"))

    def test_config_support_never_implies_training_or_restart(self):
        config = types.SimpleNamespace(model_type="qwen3")
        module = types.SimpleNamespace(AutoConfig=types.SimpleNamespace(from_pretrained=lambda _: config))
        with patch.dict("sys.modules", {"mindformers": module}):
            result = probe("qwen3")
        self.assertEqual(result["stages"]["config"], "pass")
        for stage in ("load", "train", "full_restart", "live"):
            self.assertEqual(result["stages"][stage], "not_run")


if __name__ == "__main__":
    unittest.main()
