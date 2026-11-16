"""
Configuration management module for Intelligent Cyber Threat Detection platform.
Loads and validates settings from config.yaml with environment fallback and dynamic path resolution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Recursively locates project root by looking for config.yaml or .git."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "config.yaml").exists() or (parent / ".gitignore").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT: Path = find_project_root()
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


class ConfigManager:
    """Loads, provides, and manages access to system configuration settings."""

    _instance: Optional["ConfigManager"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[Path] = None) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config(config_path or DEFAULT_CONFIG_PATH)
        return cls._instance

    def _load_config(self, config_path: Path) -> None:
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}. Falling back to default settings.")
            self._config = {}
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration successfully from {config_path}")
        except Exception as e:
            logger.error(f"Failed to parse config file at {config_path}: {e}")
            self._config = {}

    @property
    def raw(self) -> Dict[str, Any]:
        """Direct access to raw configuration dictionary."""
        return self._config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a nested configuration value using dot notation (e.g., 'dataset.target_column').
        """
        keys = key_path.split(".")
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def get_path(self, path_key: str, default: str = "") -> Path:
        """Resolves a configured path relative to the project root directory."""
        rel_path = self.get(f"paths.{path_key}", default)
        return PROJECT_ROOT / rel_path

    @property
    def raw_data_dir(self) -> Path:
        return self.get_path("raw_data_dir", "data/raw")

    @property
    def processed_data_dir(self) -> Path:
        return self.get_path("processed_data_dir", "data/processed")

    @property
    def sample_data_dir(self) -> Path:
        return self.get_path("sample_data_dir", "data/sample")

    @property
    def saved_models_dir(self) -> Path:
        return self.get_path("saved_models_dir", "models/saved_models")

    @property
    def model_metadata_dir(self) -> Path:
        return self.get_path("model_metadata_dir", "models/model_metadata")

    @property
    def reports_dir(self) -> Path:
        return self.get_path("reports_dir", "reports")

    @property
    def figures_dir(self) -> Path:
        return self.get_path("reports_figures_dir", "reports/figures")

    @property
    def results_dir(self) -> Path:
        return self.get_path("reports_results_dir", "reports/results")

    @property
    def target_column(self) -> str:
        return self.get("dataset.target_column", "Label")

    @property
    def binary_target_column(self) -> str:
        return self.get("dataset.binary_target_column", "Binary_Label")

    @property
    def benign_label(self) -> str:
        return self.get("dataset.benign_label", "BENIGN")

    @property
    def attack_label(self) -> str:
        return self.get("dataset.attack_label", "ATTACK")

    @property
    def leakage_features(self) -> List[str]:
        return self.get("dataset.leakage_features", [])

    @property
    def random_state(self) -> int:
        return int(self.get("models.random_state", 42))

    @property
    def test_size(self) -> float:
        return float(self.get("models.test_size", 0.25))


# Global convenience singleton
config = ConfigManager()

