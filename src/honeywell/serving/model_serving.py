"""Model serving module for Marvel characters."""

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from loguru import logger
from databricks.sdk.errors import ResourceDoesNotExist
import time
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput
from databricks.sdk.errors import ResourceDoesNotExist
import datetime

class ModelServing:
    """Manages model serving in Databricks for Marvel characters."""

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        """Initialize the Model Serving Manager.

        :param model_name: Name of the model to be served
        :param endpoint_name: Name of the serving endpoint
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self) -> str:
        """Retrieve the latest version of the model.

        :return: Latest version of the model as a string
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    # def deploy_or_update_serving_endpoint(
    #     self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True
    # ) -> None:
    #     """Deploy or update the model serving endpoint in Databricks for Marvel characters.

    #     :param version: Model version to serve (default: "latest")
    #     :param workload_size: Size of the serving workload (default: "Small")
    #     :param scale_to_zero: Whether to enable scale-to-zero (default: True)
        
    #     IMPORTANT:
    #     - version MUST be explicitly provided
    #     - this function must NOT resolve aliases or "latest"
    #     - this function must NOT move Champion/Challenger aliases
    #     """

    #     if not version:
    #         raise ValueError("Model version must be explicitly provided for deployment")
    #     endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())

    #     served_entities = [
    #         ServedEntityInput(
    #             entity_name=self.model_name,
    #             entity_version=str(version),  
    #             scale_to_zero_enabled=scale_to_zero,
    #             workload_size=workload_size,
    #         )
    #     ]

    #     try:
    #         # ✅ Reliable existence check
    #         self.workspace.serving_endpoints.get(self.endpoint_name)
    #         endpoint_exists = True
    #     except ResourceDoesNotExist:
    #         endpoint_exists = False

    #     if not endpoint_exists:
    #         self.workspace.serving_endpoints.create(
    #             name=self.endpoint_name,
    #             config=EndpointCoreConfigInput(
    #                 served_entities=served_entities
    #             ),
    #         )
    #         logger.info(
    #             "✅ Serving endpoint CREATED model=%s version=%s",
    #             self.model_name,
    #             version,
    #         )
    #     else:
    #         self.workspace.serving_endpoints.update_config(
    #             name=self.endpoint_name,
    #             served_entities=served_entities,
    #         )
    #         logger.info(
    #             "✅ Serving endpoint UPDATED model=%s version=%s",
    #             self.model_name,
    #             version,
    #         )



# import datetime
# from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput
# from databricks.sdk.errors import ResourceDoesNotExist, ResourceConflict

    def deploy_or_update_serving_endpoint(
        self, version: str, workload_size: str = "Small", scale_to_zero: bool = True
    ) -> None:
        
        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                entity_version=str(version),  
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
            )
        ]

        try:
            # 1. Check if it exists
            endpoint = self.workspace.serving_endpoints.get(self.endpoint_name)
            
            # 2. IF IT EXISTS: Wait for any current updates to finish BEFORE trying to update again
            # This prevents the 'ResourceConflict' error
            logger.info("Endpoint exists. Ensuring no other updates are in progress...")
            self.workspace.serving_endpoints.wait_get_serving_endpoint_not_updating(
                name=self.endpoint_name,
                timeout=datetime.timedelta(minutes=10)
            )

            # 3. Perform the UPDATE
            logger.info("Updating serving endpoint: %s to version %s...", self.endpoint_name, version)
            self.workspace.serving_endpoints.update_config(
                name=self.endpoint_name,
                served_entities=served_entities,
            )
            
            # 4. Wait for the new update to reach 'READY'
            self.workspace.serving_endpoints.wait_get_serving_endpoint_not_updating(
                name=self.endpoint_name,
                timeout=datetime.timedelta(minutes=20)
            )
            logger.info("✅ Serving endpoint UPDATED and READY.")

        except ResourceDoesNotExist:
            # 5. CREATE and WAIT
            logger.info("Creating NEW serving endpoint: %s...", self.endpoint_name)
            op = self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(served_entities=served_entities),
            )
            op.result(timeout=datetime.timedelta(minutes=20))
            logger.info("✅ Serving endpoint CREATED and READY.")

    # Set Task Values for Smoke Test
    # dbutils.jobs.taskValues.set(key="endpoint_name", value=self.endpoint_name)
    # dbutils.jobs.taskValues.set(key="workspace_url", value=self.workspace.config.host)

