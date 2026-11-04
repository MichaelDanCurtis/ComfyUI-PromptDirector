import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from prompt_director.credentials import (
    CredentialUnavailable,
    default_credentials_path,
    delete_provider_profile,
    list_provider_profiles,
    load_provider_credentials,
    save_provider_profile,
)


class CredentialProfileTests(unittest.TestCase):
    def test_default_credentials_path_can_be_overridden_for_tests_and_launchers(self):
        override = "/tmp/prompt-director-credentials.json"

        self.assertEqual(
            default_credentials_path(env={"PROMPT_DIRECTOR_CREDENTIALS_FILE": override}),
            Path(override),
        )

    def test_default_credentials_path_uses_comfyui_system_credentials_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = Path(tmpdir) / "user"
            fake_folder_paths = types.SimpleNamespace(
                get_user_directory=lambda: str(user_dir),
                get_system_user_directory=lambda name="system": str(user_dir / f"__{name}"),
            )

            previous = sys.modules.get("folder_paths")
            sys.modules["folder_paths"] = fake_folder_paths
            try:
                self.assertEqual(
                    default_credentials_path(env={}),
                    user_dir / "__credentials" / "provider_profiles.json",
                )
            finally:
                if previous is None:
                    sys.modules.pop("folder_paths", None)
                else:
                    sys.modules["folder_paths"] = previous

    def test_loads_provider_credentials_from_named_local_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "credentials.json"
            credentials_path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "default": {
                                "kimi": {
                                    "api_key": "local-secret",
                                    "base_url": "https://profile.example/v1",
                                    "model": "profile-model",
                                    "auth_style": "header:X-API-Key",
                                    "headers": {"X-Route": "vision"},
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            credentials = load_provider_credentials(
                provider_name="kimi",
                profile_name="default",
                credentials_path=credentials_path,
            )

        self.assertEqual(credentials["api_key"], "local-secret")
        self.assertEqual(credentials["base_url"], "https://profile.example/v1")
        self.assertEqual(credentials["model"], "profile-model")
        self.assertEqual(credentials["auth_style"], "header:X-API-Key")
        self.assertEqual(credentials["headers"], {"X-Route": "vision"})

    def test_shared_profile_writer_is_atomic_masked_and_preserves_existing_secret(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "provider_profiles.json"
            save_provider_profile(
                profile_name="default",
                provider_name="kimi",
                values={
                    "api_key": "secret-one",
                    "base_url": "https://api.example/v1",
                    "model": "vision-one",
                    "auth_style": "bearer",
                },
                credentials_path=path,
            )
            save_provider_profile(
                profile_name="default",
                provider_name="kimi",
                values={"api_key": "", "model": "vision-two"},
                credentials_path=path,
            )

            loaded = load_provider_credentials("kimi", credentials_path=path)
            masked = list_provider_profiles(credentials_path=path)

            self.assertEqual(loaded["api_key"], "secret-one")
            self.assertEqual(loaded["model"], "vision-two")
            self.assertTrue(masked["profiles"]["default"]["kimi"]["configured"])
            self.assertNotIn("secret-one", json.dumps(masked))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

            delete_provider_profile("default", "kimi", credentials_path=path)
            self.assertEqual(list_provider_profiles(credentials_path=path)["profiles"], {})

    def test_loads_credentials_from_legacy_prompt_director_path_when_shared_store_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = Path(tmpdir) / "user"
            credentials_path = user_dir / "prompt_director" / "credentials.json"
            credentials_path.parent.mkdir(parents=True)
            credentials_path.write_text(
                json.dumps({"profiles": {"default": {"kimi": {"api_key": "legacy-secret"}}}}),
                encoding="utf-8",
            )
            fake_folder_paths = types.SimpleNamespace(
                get_user_directory=lambda: str(user_dir),
                get_system_user_directory=lambda name="system": str(user_dir / f"__{name}"),
            )

            previous = sys.modules.get("folder_paths")
            sys.modules["folder_paths"] = fake_folder_paths
            try:
                credentials = load_provider_credentials(provider_name="kimi", profile_name="default", env={})
            finally:
                if previous is None:
                    sys.modules.pop("folder_paths", None)
                else:
                    sys.modules["folder_paths"] = previous

        self.assertEqual(credentials["api_key"], "legacy-secret")

    def test_loads_credentials_from_legacy_documented_default_user_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = Path(tmpdir) / "user"
            credentials_path = user_dir / "default" / "prompt_director" / "credentials.json"
            credentials_path.parent.mkdir(parents=True)
            credentials_path.write_text(
                json.dumps({"profiles": {"default": {"grok": {"api_key": "documented-secret"}}}}),
                encoding="utf-8",
            )
            fake_folder_paths = types.SimpleNamespace(
                get_user_directory=lambda: str(user_dir),
                get_system_user_directory=lambda name="system": str(user_dir / f"__{name}"),
            )

            previous = sys.modules.get("folder_paths")
            sys.modules["folder_paths"] = fake_folder_paths
            try:
                credentials = load_provider_credentials(provider_name="grok", profile_name="default", env={})
            finally:
                if previous is None:
                    sys.modules.pop("folder_paths", None)
                else:
                    sys.modules["folder_paths"] = previous

        self.assertEqual(credentials["api_key"], "documented-secret")

    def test_invalid_shared_credentials_file_does_not_fall_back_to_legacy_secret(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_dir = Path(tmpdir) / "user"
            shared_path = user_dir / "__credentials" / "provider_profiles.json"
            legacy_path = user_dir / "prompt_director" / "credentials.json"
            shared_path.parent.mkdir(parents=True)
            legacy_path.parent.mkdir(parents=True)
            shared_path.write_text("{not-json", encoding="utf-8")
            legacy_path.write_text(
                json.dumps({"profiles": {"default": {"kimi": {"api_key": "legacy-secret"}}}}),
                encoding="utf-8",
            )
            fake_folder_paths = types.SimpleNamespace(
                get_user_directory=lambda: str(user_dir),
                get_system_user_directory=lambda name="system": str(user_dir / f"__{name}"),
            )

            previous = sys.modules.get("folder_paths")
            sys.modules["folder_paths"] = fake_folder_paths
            try:
                with self.assertRaises(CredentialUnavailable) as caught:
                    load_provider_credentials(provider_name="kimi", profile_name="default", env={})
            finally:
                if previous is None:
                    sys.modules.pop("folder_paths", None)
                else:
                    sys.modules["folder_paths"] = previous

        self.assertIn("not valid JSON", str(caught.exception))
        self.assertNotIn("legacy-secret", str(caught.exception))

    def test_missing_local_profile_reports_actionable_error_without_secret_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "credentials.json"
            credentials_path.write_text(
                json.dumps({"profiles": {"default": {"kimi": {"api_key": "local-secret"}}}}),
                encoding="utf-8",
            )

            with self.assertRaises(CredentialUnavailable) as caught:
                load_provider_credentials(
                    provider_name="kimi",
                    profile_name="missing",
                    credentials_path=credentials_path,
                )

        self.assertIn("missing", str(caught.exception))
        self.assertIn(str(credentials_path), str(caught.exception))
        self.assertNotIn("local-secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
