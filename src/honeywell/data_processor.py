"""Data preprocessing module for EV battery charging data."""

import time
import uuid
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp, lit
from sklearn.model_selection import train_test_split

from honeywell.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing EV battery charging DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the EV battery charging DataFrame stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering.
        """
        cat_features = self.config.cat_features
        num_features = self.config.num_features
        target = self.config.target
        print(list(self.df.columns))

        self.df.rename(columns={"SOC (%)": "SOC"}, inplace=True)
        self.df.rename(columns={"Voltage (V)": "Voltage"}, inplace=True)
        self.df.rename(columns={"Current (A)": "Current"}, inplace=True)
        self.df.rename(columns={"Battery Temp (°C)": "Battery_Temp"}, inplace=True)
        self.df.rename(columns={"Ambient Temp (°C)": "Ambient_Temp"}, inplace=True)
        self.df.rename(columns={"Charging Duration (min)": "Charging_Duration"}, inplace=True)
        self.df.rename(columns={"Degradation Rate (%)": "Degradation_Rate"}, inplace=True)
        self.df.rename(columns={"Efficiency (%)": "Efficiency"}, inplace=True)
        self.df.rename(columns={"Charging Cycles": "Charging_Cycles"}, inplace=True)
        self.df.rename(columns={"EV Model": "EV_Model"}, inplace=True)
        self.df.rename(columns={"Optimal Charging Duration Class": "Optimal_Charging_Duration_Class"}, inplace=True)
        self.df.rename(columns={"Battery Type": "Battery_Type"}, inplace=True)
        self.df.rename(columns={"Charging Mode": "Charging_Mode"}, inplace=True)

        print(list(self.df.columns),"2222222")
        # GOOD: Replaces only the missing values with -1.0, keeps other numbers as they are
        self.df["SOC"] = self.df["SOC"].fillna("-1")
        self.df["Voltage"] = self.df["Voltage"].fillna(-1.0)
        self.df["Current"] = self.df["Current"].fillna(-1.0)
        self.df["Battery_Temp"] = self.df["Battery_Temp"].fillna(-1.0)
        self.df["Ambient_Temp"] = self.df["Ambient_Temp"].fillna(-1.0)
        self.df["Charging_Duration"] = self.df["Charging_Duration"].fillna(-1.0)
        self.df["Degradation_Rate"] = self.df["Degradation_Rate"].fillna(-1.0)
        self.df["Efficiency"] = self.df["Efficiency"].fillna(-1.0)  
        self.df["Battery_Type"] = self.df["Battery_Type"].fillna("Unknown")
        self.df["Charging_Cycles"] = self.df["Charging_Cycles"].fillna(-1.0)
        self.df["Charging_Mode"] = self.df["Charging_Mode"].fillna("Unknown")
        self.df["EV_Model"] = self.df["EV_Model"].fillna("Unknown")
        self.df["Optimal_Charging_Duration_Class"] = self.df["Optimal_Charging_Duration_Class"].fillna(-1.0)


        # 1. Keep only rows that have 0, 1, or 2 (Cleaning)
        self.df = self.df[self.df["Optimal_Charging_Duration_Class"].astype(str).isin(["0", "1", "2"])]

        # 2. Convert the column to integers so LightGBM understands it's a numeric target
        self.df["Optimal_Charging_Duration_Class"] = self.df["Optimal_Charging_Duration_Class"].astype(int)

        self.df = self.df[num_features + cat_features + [target] ]

        for col in cat_features:
            self.df[col] = self.df[col].astype("category")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    # def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
    #     """Save the train and test sets into Databricks tables.

    #     :param train_set: The training DataFrame to be saved.
    #     :param test_set: The test DataFrame to be saved.
    #     """
    #     train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
    #         "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
    #     )

    #     test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
    #         "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
    #     )

    #     train_set_with_timestamp.write.mode("overwrite").saveAsTable(
    #         f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
    #     )

    #     test_set_with_timestamp.write.mode("overwrite").saveAsTable(
    #         f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
    #     )

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        # 1. Create a Unique Run ID for this specific training session
        current_run_id = str(uuid.uuid4())[:8] 
        
        # 2. Add Timestamp AND Run ID to the data
        train_df = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        ).withColumn("run_id", lit(current_run_id))

        test_df = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        ).withColumn("run_id", lit(current_run_id))

        # 3. Use APPEND mode to keep old data + add new data
        train_df.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_df.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

