# COMMAND ----------

import os
import mlflow
import yaml
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from honeywell.config import GitTagsFromWidgets, ProjectConfig, Tags
from honeywell.models.basic_model import BasicModel
from pyspark.sql import SparkSession

from honeywell import BasicModel
from dotenv import load_dotenv

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# COMMAND ----------
# Set up Databricks or local MLflow tracking
def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../project_config_honeywell.yml", env="dev")
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

git_tags = GitTagsFromWidgets.from_widgets(dbutils)
logger.info("Git tags loaded:")
tags = Tags(**{
    "git_sha": git_tags.git_sha,
    "branch": git_tags.branch
})
# COMMAND ----------
# Initialize model with the config path
basic_model = BasicModel(config=config,
                         tags=tags,
                         spark=spark)
# COMMAND ----------
basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------
basic_model.train()

# COMMAND ----------
run_id = basic_model.log_model()

# COMMAND ----------
dbutils.jobs.taskValues.set("candidate_run_id", run_id)

# COMMAND ----------
basic_model.register_model()
