# COMMAND ----------
import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from honeywell.config import ProjectConfig
# from databricks-cicd.data_processor import DataProcessor

import seaborn as sns 
import matplotlib.pyplot as plt

config = ProjectConfig.from_yaml(config_path="../project_config_honeywell.yml", env="dev")
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
df = pd.read_csv(config.input_data_path)
logger.info(f"Data loaded from {config.input_data_path} with shape {df.shape}")
df.head()

# COMMAND ----------
df.shape

# COMMAND ----------
df.describe()

# COMMAND ----------
df.info()

# COMMAND ----------
df.isnull().sum()

# COMMAND ----------
df.drop(columns="device_id", inplace=True,axis=1)

# COMMAND ----------
df.head()

# COMMAND ----------
plt.figure(figsize = (12 , 5))
sns.lineplot(data = df , x = 'daily_usage_hours' , y = 'performance_rating')
plt.show()

# COMMAND ----------
plt.figure(figsize=(12, 5))
sns.set_theme(context = 'talk' , style = 'whitegrid')
sns.lineplot(data = df , x = 'battery_health_percent' , y = 'performance_rating' , markers = '-' , color = 'blue' , linewidth = 2)
plt.title('Performance Rating / Battery_Health_Percent')
plt.show()

# COMMAND ----------
plt.figure(figsize=(16, 7))
sns.scatterplot(data = df , x = 'battery_health_percent' , y = 'battery_age_months' , hue = 'model_year')
plt.show()

# COMMAND ----------
from sklearn.model_selection import train_test_split
X = df.drop(columns = 'performance_rating' , axis = 1)
y = df['performance_rating']
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)

# COMMAND ----------
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# COMMAND ----------
categorical_cols = [col for col in df.columns if df[col].dtype == 'O' and col != 'overheating_issues']

for col in categorical_cols : 
    print(df[col].unique())

# COMMAND ----------
X_train['overheating_issues'] = X_train['overheating_issues'].astype(bool)
X_test['overheating_issues'] = X_test['overheating_issues'].astype(bool)

# COMMAND ----------
transformer = ColumnTransformer(transformers = [ 
    ('one_hot' , OneHotEncoder(drop = 'first' , handle_unknown = 'ignore' , dtype = int) , categorical_cols)
] , remainder = 'passthrough')

# COMMAND ----------

X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

column_name = transformer.get_feature_names_out()

X_train_transformed = pd.DataFrame(data = X_train_transformed , columns = column_name , dtype = int)
X_test_transformed = pd.DataFrame(data = X_test_transformed , columns = column_name , dtype = int)


X_train_transformed.head()

# COMMAND ----------
X_train_transformed.shape

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor , GradientBoostingRegressor
from xgboost import XGBRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgbm

# COMMAND ----------
models = {
    'RFR' : RandomForestRegressor(),
    'ABR' : AdaBoostRegressor(),
    'GBR' : GradientBoostingRegressor(),
    'DTR' : DecisionTreeRegressor(),
    'KNN' : KNeighborsRegressor(),
    'XGBR' : XGBRegressor(),
    'LGBM': lgbm.LGBMRegressor(verbose=0, 
                           min_child_samples=2,  # Varsayılan 20'dir, 2'ye düşürün
                           min_data_in_bin=3)
}

# COMMAND ----------
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
def calculate_metrics(true, predict) : 
        
    r2 = r2_score(true , predict)
    mae = mean_absolute_error(true , predict)
    mse = mean_squared_error(true , predict)
 
    return r2 , mae , mse

# COMMAND ----------

for i in range(len(models)) : 
    model = list(models.values())[i]
    model.fit(X_train_transformed , y_train)

    y_pred = model.predict(X_test_transformed)

    r2 , mae , mse = calculate_metrics(y_test , y_pred)

    print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n')
    print(list(models.keys())[i])
    print(' R2 :  '  , r2)
    print(' Mean Absolute Error : '  , mae)
    print(' Mean Squared Error : '  , mse)

# COMMAND ----------
import pickle
with open('../models/transformer.pkl' , 'wb') as f :
    pickle.dump(transformer , f)    
print('Transformer saved successfully!')
# COMMAND ----------
