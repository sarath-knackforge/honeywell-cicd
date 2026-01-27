from typing import Any

# from typing import String
import yaml
from pydantic import BaseModel
import subprocess
from typing import Optional, Any

class ProjectConfig(BaseModel):
    """Represent project configuration parameters loaded from YAML.

    Handles feature specifications, catalog details, and experiment parameters.
    Supports environment-specific configuration overrides.
    """

    num_features: list[str]
    cat_features: list[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: dict[str, Any]
    experiment_name_basic: str | None
    experiment_name_custom: str | None
    input_data_path: str | None

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load and parse configuration settings from a YAML file.

        :param config_path: Path to the YAML configuration file
        :param env: Environment name to load environment-specific settings
        :return: ProjectConfig instance initialized with parsed configuration
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]
            config_dict["input_data_path"] = config_dict["input_data_path"]
            config_dict["target"] = config_dict["target"]

            return cls(**config_dict)

class Tags(BaseModel):
    """Model for MLflow tags."""

    git_sha: str
    branch: str
    run_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert the Tags instance to a dictionary."""
        tags_dict = {}
        tags_dict["git_sha"] = self.git_sha
        tags_dict["branch"] = self.branch
        if self.run_id is not None:
            tags_dict["run_id"] = self.run_id
        return tags_dict


class GitTagsFromWidgets(BaseModel):
    git_sha: str
    branch: str

    @classmethod
    def from_widgets(cls, dbutils: Optional[str] = None) -> "GitTagsFromWidgets":
        # 1) Try Databricks widgets
        try:
            dbutils.widgets.text("git_sha", "")
            dbutils.widgets.text("branch", "")

            git_sha = (dbutils.widgets.get("git_sha") or "").strip()
            branch = (dbutils.widgets.get("branch") or "").strip()

            if git_sha or branch:
                return cls(
                    git_sha=git_sha or "unknown",
                    branch=branch or "unknown"
                )

        except Exception:
            pass

        # 2) Fallback: try local git (for dev only)
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            return cls(
                git_sha=git_sha,
                branch=branch
            )

        except Exception:
            # 3) Final fallback
            return cls(
                git_sha="unknown",
                branch="unknown"
            )