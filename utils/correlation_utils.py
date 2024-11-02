import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_correlation(df, target_columns):
    """
    Calculates the correlation of all features with the target columns.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing features and targets.
    - target_columns (list): List of target column names.
    
    Returns:
    - pd.DataFrame: Correlation matrix.
    """
    correlation_matrix = df.corr().loc[target_columns]
    return correlation_matrix

def calculate_vif(df):
    """
    Calculates Variance Inflation Factor (VIF) for each feature.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing features.
    
    Returns:
    - pd.DataFrame: VIF scores for each feature.
    """
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty. Please check your data.")

    # Check if there are numeric columns
    numeric_columns = check_numeric_columns(df)
    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns found in the DataFrame.")

    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_columns
    vif_data["VIF"] = [variance_inflation_factor(df[numeric_columns].values, i) for i in range(len(numeric_columns))]
    return vif_data

def check_numeric_columns(df):
    """
    Checks and returns numeric columns in the dataframe.
    
    Parameters:
    - df (pd.DataFrame): The dataframe to check.
    
    Returns:
    - list: List of numeric column names.
    """
    return df.select_dtypes(include=['number']).columns.tolist()