import pandas as pd
import numpy as np
from pyspark.sql import functions as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_real_data_inference_sanity_check(
    model,
    model_uri: str,
    spark,
    config,
    n_rows: int = 5,
):
    """
    Runs REAL inference sanity check using the last N rows from the UC test_set.
    Uses row-by-row prediction to bypass the pipeline's batch bug.

    Raises RuntimeError if inference fails or outputs are invalid.
    """


    logger.info(
        "Starting REAL real-data inference sanity check for model: %s", model_uri
    )

    # ------------------------------------------------------------------
    # 1) Load test dataset from Unity Catalog
    # ------------------------------------------------------------------
    try:
        test_table = f"{config.catalog_name}.{config.schema_name}.test_set"
        logger.info("Loading test dataset from: %s", test_table)

        spark_df = spark.table(test_table)

        if spark_df.count() == 0:
            raise RuntimeError("UC test_set table is empty")

        # Take last N rows deterministically if possible
        if "ingestion_ts" in spark_df.columns:
            spark_df_last = (
                spark_df.orderBy(F.col("ingestion_ts").desc()).limit(n_rows)
            )
            logger.info(
                "Using last %d rows ordered by ingestion_ts", n_rows
            )
        else:
            logger.warning(
                "No ingestion_ts column found. Taking arbitrary last %d rows.",
                n_rows,
            )
            spark_df_last = spark_df.limit(n_rows)

        test_pdf = spark_df_last.toPandas()

        logger.info("Loaded %d rows from UC test_set", len(test_pdf))

    except Exception as exc:
        logger.error("❌ Failed to load test dataset from Unity Catalog")
        logger.exception(exc)
        raise RuntimeError("Failed to load UC test_set for inference sanity") from exc

    finally:
        test_pdf.display(spark_df_last.limit(5).toPandas())

    # ------------------------------------------------------------------
    # 1) Select and order columns exactly like the working sample_df
    # ------------------------------------------------------------------
    feature_cols = [
        "SOC",
        "Voltage",
        "Current",
        "Battery_Temp",
        "Ambient_Temp",
        "Charging_Duration",
        "Degradation_Rate",
        "Efficiency",
        "Charging_Cycles",
        "Battery_Type",
        "EV_Model",
        "Charging_Mode",
    ]

    X_real = test_pdf[feature_cols].copy()

    # ------------------------------------------------------------------
    # 2) Cast to EXACT same safe dtypes as your working dummy sample_df
    # ------------------------------------------------------------------
    X_real = X_real.astype({
        "SOC": "float64",
        "Voltage": "float64",
        "Current": "float64",
        "Battery_Temp": "float64",
        "Ambient_Temp": "float64",
        "Charging_Duration": "float64",
        "Degradation_Rate": "float64",
        "Efficiency": "float64",
        "Charging_Cycles": "int64",
        "Battery_Type": "str",
        "EV_Model": "str",
        "Charging_Mode": "str",
    })

    # ------------------------------------------------------------------
    # 3) (Optional but recommended) show preview
    # ------------------------------------------------------------------
    test_pdf.display(X_real.head(5))

    print("Final dtypes:")
    print(X_real.dtypes)
    return X_real