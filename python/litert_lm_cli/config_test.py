# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the LiteRT-LM CLI configuration manager."""

import json
import os

from absl.testing import absltest
from absl.testing import parameterized
import click

from litert_lm_cli import config


class ConfigTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config._clear_cache()
    # Save original env
    self._original_env = os.environ.get("LITERT_LM_DIR")
    # Create temp dir and set env
    self.temp_dir = self.create_tempdir()
    os.environ["LITERT_LM_DIR"] = self.temp_dir.full_path

  def tearDown(self):
    # Restore original env
    if self._original_env is not None:
      os.environ["LITERT_LM_DIR"] = self._original_env
    else:
      os.environ.pop("LITERT_LM_DIR", None)
    super().tearDown()

  def _write_config(self, content: str) -> None:
    """Helper to write content to the config.json file."""
    config_path = config.get_config_path()
    with open(config_path, "w") as f:
      f.write(content)

  def test_get_config_no_file(self):
    # No file exists in the temp dir
    self.assertEqual(config.load_config(), config.AppConfig())

  def test_get_config_valid(self):
    self._write_config(
        '{"default": {"backend": "gpu", "cpu_thread_count": 4}, "models":'
        ' {"m1": {"cpu_thread_count": 8}}}'
    )
    self.assertEqual(
        config.load_config(),
        config.AppConfig(
            default=config.ModelConfig(backend="gpu", cpu_thread_count=4),
            models={
                "m1": config.ModelConfig(cpu_thread_count=8),
            },
        ),
    )

  def test_get_config_invalid_json(self):
    self._write_config("invalid json")
    with self.assertRaises(click.ClickException) as ctx:
      config.load_config()
    self.assertIn("Failed to parse config.json", str(ctx.exception))

  @parameterized.named_parameters(
      ("not_dict", "[]", "Config must be a JSON object"),
      (
          "default_not_dict",
          '{"default": 123}',
          "default: 123 is not of type 'object'",
      ),
      (
          "backend_invalid",
          '{"default": {"backend": "invalid"}}',
          "default.backend: 'invalid' is not one of ['cpu', 'gpu', 'npu']",
      ),
      (
          "default_cpu_thread_count_not_int",
          '{"default": {"cpu_thread_count": "four"}}',
          "default.cpu_thread_count: 'four' is not of type 'integer'",
      ),
      (
          "default_cpu_thread_count_invalid",
          '{"default": {"cpu_thread_count": 0}}',
          "default.cpu_thread_count: 0 is less than the minimum of 1",
      ),
      (
          "default_cpu_thread_count_negative",
          '{"default": {"cpu_thread_count": -1}}',
          "default.cpu_thread_count: -1 is less than the minimum of 1",
      ),
      (
          "models_not_dict",
          '{"models": []}',
          "models: [] is not of type 'object'",
      ),
      (
          "model_entry_not_dict",
          '{"models": {"m": 123}}',
          "models.m: 123 is not of type 'object'",
      ),
      (
          "model_backend_invalid",
          '{"models": {"m": {"backend": "invalid"}}}',
          "models.m.backend: 'invalid' is not one of ['cpu', 'gpu', 'npu']",
      ),
      (
          "model_cpu_thread_count_not_int",
          '{"models": {"m": {"cpu_thread_count": "four"}}}',
          "models.m.cpu_thread_count: 'four' is not of type 'integer'",
      ),
      (
          "model_cpu_thread_count_invalid",
          '{"models": {"m": {"cpu_thread_count": 0}}}',
          "models.m.cpu_thread_count: 0 is less than the minimum of 1",
      ),
  )
  def test_get_config_invalid_schema(self, json_data, expected_error):
    self._write_config(json_data)
    with self.assertRaises(click.ClickException) as ctx:
      config.load_config()
    self.assertIn(expected_error, str(ctx.exception))

  def test_get_model_config_no_file(self):
    result = config.get_model_config("my-model")
    self.assertIsInstance(result, config.ModelConfig)
    self.assertIsNone(result.backend)

  def test_get_model_config_valid(self):
    self._write_config('{"models": {"my-model": {"backend": "gpu"}}}')
    result = config.get_model_config("my-model")
    self.assertIsInstance(result, config.ModelConfig)
    self.assertEqual(result.backend, "gpu")

  def test_get_model_config_not_configured(self):
    self._write_config('{"models": {"other-model": {"backend": "gpu"}}}')
    result = config.get_model_config("my-model")
    self.assertIsInstance(result, config.ModelConfig)
    self.assertIsNone(result.backend)

  def test_get_model_config_with_fallback(self):
    self._write_config(
        '{"default": {"backend": "cpu", "cpu_thread_count": 4}, "models":'
        ' {"gpu-model": {"backend": "gpu"}, "custom-cpu-model":'
        ' {"cpu_thread_count": 8}, "empty-model": {}}}'
    )

    # 1. Model not in config -> should fall back to default (cpu, 4 threads)
    result1 = config.get_model_config("not-configured-model")
    self.assertEqual(result1.backend, "cpu")
    self.assertEqual(result1.cpu_thread_count, 4)

    # 2. Model in config with backend (gpu) only -> should use model backend
    # (gpu) and default threads (4)
    result2 = config.get_model_config("gpu-model")
    self.assertEqual(result2.backend, "gpu")
    self.assertEqual(result2.cpu_thread_count, 4)

    # 3. Model in config with threads (8) only -> should fall back to default
    # backend (cpu) and use model threads (8)
    result3 = config.get_model_config("custom-cpu-model")
    self.assertEqual(result3.backend, "cpu")
    self.assertEqual(result3.cpu_thread_count, 8)

    # 4. Model in config but empty -> should fall back to default (cpu, 4
    # threads)
    result4 = config.get_model_config("empty-model")
    self.assertEqual(result4.backend, "cpu")
    self.assertEqual(result4.cpu_thread_count, 4)


if __name__ == "__main__":
  absltest.main()
