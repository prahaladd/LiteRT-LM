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
"""Configuration manager for LiteRT-LM CLI."""

from __future__ import annotations

import dataclasses
import json
import os
import pkgutil
from typing import Any

import click
import jsonschema

KEY_DEFAULT = "default"
KEY_MODELS = "models"
KEY_BACKEND = "backend"
KEY_CPU_THREAD_COUNT = "cpu_thread_count"


@dataclasses.dataclass
class ModelConfig:
  backend: str | None = None
  cpu_thread_count: int | None = None

  def merge_from(self, other: ModelConfig) -> None:
    """Merges values from another config, overwriting if set."""
    if other.backend is not None:
      self.backend = other.backend
    if other.cpu_thread_count is not None:
      self.cpu_thread_count = other.cpu_thread_count


@dataclasses.dataclass
class AppConfig:
  default: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  models: dict[str, ModelConfig] = dataclasses.field(default_factory=dict)


_CACHED_CONFIG: AppConfig | None = None
_SCHEMA: dict[str, Any] | None = None


def get_cli_base_dir() -> str:
  """Gets the base directory for LiteRT-LM CLI."""
  env_override = os.environ.get("LITERT_LM_DIR")
  if env_override:
    return os.path.abspath(env_override)
  return os.path.join(os.path.expanduser("~"), ".litert-lm")


def get_config_path() -> str:
  """Gets the path to the config.json file."""
  return os.path.join(get_cli_base_dir(), "config.json")


def _load_schema() -> dict[str, Any]:
  """Loads the JSON schema for configuration validation."""
  global _SCHEMA
  if _SCHEMA is not None:
    return _SCHEMA
  try:
    data = pkgutil.get_data(__name__, "config_schema.json")
    if data is not None:
      _SCHEMA = json.loads(data.decode("utf-8"))
      return _SCHEMA
  except Exception as e:
    raise click.ClickException(f"Failed to load config schema: {e}") from e

  raise click.ClickException(
      "Config schema file 'config_schema.json' not found."
  )


def _parse_model_config(data: dict[str, Any]) -> ModelConfig:
  """Parses a ModelConfig from a dict."""
  return ModelConfig(
      backend=data.get(KEY_BACKEND),
      cpu_thread_count=data.get(KEY_CPU_THREAD_COUNT),
  )


def load_config() -> AppConfig:
  """Loads and validates the config.json file."""
  global _CACHED_CONFIG
  if _CACHED_CONFIG is not None:
    return _CACHED_CONFIG

  config_path = get_config_path()
  if not os.path.exists(config_path):
    return AppConfig()

  try:
    with open(config_path, "r") as f:
      config_data = json.load(f)
  except json.JSONDecodeError as e:
    raise click.ClickException(f"Failed to parse config.json: {e}") from e
  except Exception as e:
    raise click.ClickException(f"Failed to read config.json: {e}") from e

  if not isinstance(config_data, dict):
    raise click.ClickException(
        "config.json: Config must be a JSON object (dict)."
    )

  schema = _load_schema()
  try:
    jsonschema.validate(instance=config_data, schema=schema)
  except jsonschema.ValidationError as e:
    path_str = ".".join(str(p) for p in e.path)
    prefix = f"{path_str}: " if path_str else ""
    raise click.ClickException(
        f"config.json validation error: {prefix}{e.message}"
    ) from e

  app_config = AppConfig()
  if KEY_DEFAULT in config_data:
    app_config.default = _parse_model_config(config_data[KEY_DEFAULT])
  if KEY_MODELS in config_data:
    for model_id, model_data in config_data[KEY_MODELS].items():
      app_config.models[model_id] = _parse_model_config(model_data)

  _CACHED_CONFIG = app_config
  return app_config


def get_model_config(model_id: str) -> ModelConfig:
  """Returns the configuration for a specific model, falling back to defaults."""
  config_data = load_config()
  merged = ModelConfig()

  merged.merge_from(config_data.default)

  model_cfg = config_data.models.get(model_id)
  if model_cfg is not None:
    merged.merge_from(model_cfg)

  return merged


def _clear_cache() -> None:
  """Clears the cached configuration (primarily for testing)."""
  global _CACHED_CONFIG
  _CACHED_CONFIG = None
