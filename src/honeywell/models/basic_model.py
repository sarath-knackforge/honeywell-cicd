"""Basic model implementation for EV battery charging classification.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict (Alive).
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import mlflow
import pandas as pd
import numpy as np
from delta.tables import DeltaTable
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from honeywell.config import ProjectConfig, Tags
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score, log_loss, precision_score, recall_score, roc_auc_score


class BasicModel:
    """A basic model class for EV battery charging prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.ev_battery_charging_model_basic"
        self.tags = tags.to_dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_set = self.test_set_spark.toPandas()

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        self.eval_data = self.test_set[self.num_features + self.cat_features + [self.target]]

        train_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_data_version = str(train_delta_table.history().select("version").first()[0])
        test_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_data_version = str(test_delta_table.history().select("version").first()[0])
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and LightGBM classification model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")

        class CatToIntTransformer(BaseEstimator, TransformerMixin):
            """Transformer that encodes categorical columns as integer codes for LightGBM.

            Unknown categories at transform time are encoded as -1.
            """

            def __init__(self, cat_features: list[str]) -> None:
                """Initialize the transformer with categorical feature names."""
                self.cat_features = cat_features
                self.cat_maps_ = {}

            def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
                """Fit the transformer to the DataFrame X."""
                self.fit_transform(X)
                return self

            def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
                """Fit and transform the DataFrame X."""
                X = X.copy()
                for col in self.cat_features:
                    c = pd.Categorical(X[col])
                    # Build mapping: {category: code}
                    self.cat_maps_[col] = dict(zip(c.categories, range(len(c.categories)), strict=False))
                    X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
                return X

            def transform(self, X: pd.DataFrame) -> pd.DataFrame:
                """Transform the DataFrame X by encoding categorical features as integers."""
                X = X.copy()
                for col in self.cat_features:
                    X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
                return X

        preprocessor = ColumnTransformer(
            transformers=[("cat", CatToIntTransformer(self.cat_features), self.cat_features)], remainder="passthrough"
        )
        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", LGBMClassifier(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self) -> None:
        """Log model, datasets, parameters, and metrics to MLflow.
        Supports both binary and multiclass classification.
        """

        logger.info("📦 Logging model + parameters + metrics to MLflow...")

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name="ev-battery_charing-lgbm", tags=self.tags) as run:
            # --- Run metadata ---
            self.run_id = run.info.run_id
            # try:
            #     dbutils.jobs.taskValues.set("candidate_run_id", self.run_id)
            # except Exception:
            #     pass

            # --- 1) Log parameters ---
            mlflow.log_params(self.config.parameters)

            # --- 2) Log dataset lineage ---
            train_dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.train_data_version,
            )
            mlflow.log_input(train_dataset, context="training")

            test_dataset = mlflow.data.from_spark(
                self.test_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.test_set",
                version=self.test_data_version,
            )
            mlflow.log_input(test_dataset, context="testing")

            # --- 3) Log model artifact + signature ---
            signature = infer_signature(
                model_input=self.X_train,
                model_output=self.pipeline.predict(self.X_train)
            )

            self.model_info = mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lightgbm-pipeline-model",
                signature=signature,
                input_example=self.X_test[0:1],
            )

            # --- 4) Predictions ---
            y_pred = self.pipeline.predict(self.X_test)

            # Some models (binary only) may not support predict_proba
            try:
                y_proba = self.pipeline.predict_proba(self.X_test)
            except Exception:
                y_proba = None

            # --- 5) Detect binary vs multiclass ---
            unique_classes = np.unique(self.y_test)
            num_classes = len(unique_classes)

            # For multiclass use "weighted" to handle imbalance (best for EV data)
            strategy = "binary" if num_classes == 2 else "weighted"

            # --- 6) Compute metrics ---
            metrics = {
                "accuracy": float(accuracy_score(self.y_test, y_pred)),
                "precision": float(precision_score(self.y_test, y_pred, average=strategy)),
                "recall": float(recall_score(self.y_test, y_pred, average=strategy)),
                "f1_score": float(f1_score(self.y_test, y_pred, average=strategy)),
                "num_classes": float(num_classes),
            }

            if y_proba is not None:
                metrics["log_loss"] = float(log_loss(self.y_test, y_proba))

            # --- 7) Compute ROC-AUC safely ---
                roc_auc = None
                try:
                    if num_classes == 2:
                        # Binary: use positive class probability
                        roc_auc = roc_auc_score(self.y_test, y_proba[:, 1])
                    else:
                        # Multiclass: use OvR + weighted averaging
                        roc_auc = roc_auc_score(
                            self.y_test,
                            y_proba,
                            multi_class="ovr",
                            average="weighted",
                        )
                except Exception as e:
                    roc_auc = None

            metrics["roc_auc"] = float(roc_auc) if roc_auc is not None else None
            self.metrics = metrics
            mlflow.log_metrics(metrics)

            # --- 7) Classification report artifact ---
            target_names = ["Short", "Medium", "Long"] if num_classes == 3 else None

            report = classification_report(
                self.y_test,
                y_pred,
                target_names=target_names
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"classification_report_{timestamp}.txt"

            with open(report_path, "w") as f:
                f.write(report)

            mlflow.log_artifact(report_path)

            # --- 8) Confusion matrix artifact ---
            fig, ax = plt.subplots(figsize=(8, 6))

            ConfusionMatrixDisplay.from_predictions(
                self.y_test,
                y_pred,
                display_labels=target_names,
                cmap="Blues",
                ax=ax,
            )

            cm_filename = f"confusion_matrix_{timestamp}.png"
            plt.title("Charging Duration Confusion Matrix")
            plt.tight_layout()
            plt.savefig(cm_filename)
            plt.close(fig)

            mlflow.log_artifact(cm_filename)

            logger.info("✅ Model, metrics, datasets, and artifacts logged to MLflow.")
            return self.run_id

    # def log_model(self) -> None:
    #     """Log the model using MLflow."""
    #     mlflow.set_experiment(self.experiment_name)
    #     with mlflow.start_run(tags=self.tags) as run:
    #         self.run_id = run.info.run_id

    #         signature = infer_signature(model_input=self.X_train, model_output=self.pipeline.predict(self.X_train))
    #         train_dataset = mlflow.data.from_spark(
    #             self.train_set_spark,
    #             table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
    #             version=self.train_data_version,
    #         )
    #         mlflow.log_input(train_dataset, context="training")
    #         test_dataset = mlflow.data.from_spark(
    #             self.test_set_spark,
    #             table_name=f"{self.catalog_name}.{self.schema_name}.test_set",
    #             version=self.test_data_version,
    #         )
    #         mlflow.log_input(test_dataset, context="testing")
    #         self.model_info = mlflow.sklearn.log_model(
    #             sk_model=self.pipeline,
    #             artifact_path="lightgbm-pipeline-model",
    #             signature=signature,
    #             input_example=self.X_test[0:1],
    #         )
    #         eval_data = self.X_test.copy()
    #         eval_data[self.config.target] = self.y_test

    #         result = mlflow.models.evaluate(
    #             self.model_info.model_uri,
    #             eval_data,
    #             targets=self.config.target,
    #             model_type="classifier",
    #             evaluators=["default"],
    #         )
    #         self.metrics = result.metrics

    # def model_improved(self) -> bool:
    #     """Evaluate the model performance on the test set.

    #     Compares the current model with the latest registered model using F1-score.
    #     :return: True if the current model performs better, False otherwise.
    #     """
    #     client = MlflowClient()
    #     latest_model_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model")
    #     latest_model_uri = f"models:/{latest_model_version.model_id}"

    #     result = mlflow.models.evaluate(
    #         latest_model_uri,
    #         self.eval_data,
    #         targets=self.config.target,
    #         model_type="classifier",
    #         evaluators=["default"],
    #     )
    #     metrics_old = result.metrics
    #     if self.metrics["f1_score"] >= metrics_old["f1_score"]:
    #         logger.info("Current model performs better. Returning True.")
    #         return True
    #     else:
    #         logger.info("Current model does not improve over latest. Returning False.")
    #         return False

    def model_improved(self) -> bool:
        """Compare current model with Champion using F1-score."""

        client = MlflowClient()

        # 1) Bootstrap: first model ever → auto-promote
        try:
            champion_version = client.get_model_version_by_alias(
                name=self.model_name,
                alias="champion"
            )
        except Exception:
            logger.info("ℹ️ No existing Champion model found. Treating this as first model.")
            return True

        champion_uri = f"models:/{champion_version.model_id}"

        # 2) Evaluate Champion on current test set
        result = mlflow.models.evaluate(
            champion_uri,
            self.eval_data,
            targets=self.config.target,
            model_type="classifier",
            evaluators=["default"],
        )
        metrics_old = result.metrics

        # 3) Robust F1 fetch
        def _get_f1(metrics: dict) -> float:
            for key in ["f1_score", "f1_score_weighted", "f1_score_macro"]:
                if key in metrics:
                    return metrics[key]
            raise KeyError("No F1-score metric found in MLflow metrics")

        f1_new = _get_f1(self.metrics)
        f1_old = _get_f1(metrics_old)

        logger.info(f"📊 F1 comparison → new: {f1_new:.4f}, champion: {f1_old:.4f}")

        if f1_new >= f1_old:
            logger.info("🏆 Current model performs better or equal. Returning True.")
            return True
        else:
            logger.info("📉 Current model does not improve over Champion. Returning False.")
            return False


    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version

 # for advanced Champion/Challenger logic (disabled)

    # def register_model(self) -> str:
    #     """Register model in Unity Catalog with Champion/Challenger logic."""
    #     logger.info("🔄 Registering the model in UC...")

    #     client = MlflowClient()

    #     # 1) Register the new model version
    #     registered_model = mlflow.register_model(
    #         model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
    #         name=self.model_name,
    #         tags=self.tags,
    #     )

    #     new_version = registered_model.version
    #     logger.info(f"✅ Model registered as version {new_version}.")

    #     # 2) Decide if the model is better than the current Champion
    #     try:
    #         is_better = self.model_improved()
    #     except Exception as e:
    #         logger.warning(
    #             "⚠️ Could not compare with existing Champion model. "
    #             "Assuming this is the first model or comparison failed.",
    #             exc_info=True
    #         )
    #         is_better = True   # first model → make it Champion

    #     # 3) Set aliases based on comparison
    #     if is_better:
    #         logger.info("🏆 New model is better. Promoting to Champion.")

    #         # Move old Champion → Challenger (if exists)
    #         try:
    #             current_champion = client.get_model_version_by_alias(
    #                 name=self.model_name,
    #                 alias="champion"
    #             )
    #             client.set_registered_model_alias(
    #                 name=self.model_name,
    #                 alias="challenger",
    #                 version=current_champion.version
    #             )
    #             logger.info(
    #                 f"♻️ Previous Champion v{current_champion.version} "
    #                 f"moved to Challenger."
    #             )
    #         except Exception:
    #             logger.info("ℹ️ No existing Champion found. Skipping demotion step.")

    #         # Set new Champion
    #         client.set_registered_model_alias(
    #             name=self.model_name,
    #             alias="champion",
    #             version=new_version
    #         )

    #         # (Optional) keep latest-model in sync with Champion
    #         client.set_registered_model_alias(
    #             name=self.model_name,
    #             alias="latest-model",
    #             version=new_version
    #         )

    #         return "champion"

    #     else:
    #         logger.info("🧪 New model is NOT better. Marking as Challenger.")

    #         client.set_registered_model_alias(
    #             name=self.model_name,
    #             alias="challenger",
    #             version=new_version
    #         )

    #         return "challenger"

