# correlation_analysis.py

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from io import BytesIO
from utils.correlation_utils import calculate_correlation, calculate_vif, check_numeric_columns
import pickle
from utils.database import save_model, get_saved_models, load_model
from utils.sports_data import fetch_sport_data

# Title of the page
st.title("Key Stat Analysis")
st.markdown("Uncover which stats impact game outcomes the most. Upload your data or use built-in options to analyze stat correlations and feature importance. The visual heatmaps and weighted formulas help you understand what drives performance.")

# General Styling
st.markdown("""
    <style>
        /* Shared CSS for consistent styling */
        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
            background: var(--background-color);
            color: var(--text-color);
        }

        :root {
            --background-color: #2C3E50; /* Charcoal Dark Gray */
            --primary-color: #1E90FF; /* Electric Blue */
            --secondary-color: #FF8C00; /* Deep Orange */
            --accent-color: #FF4500; /* Fiery Red */
            --success-color: #32CD32; /* Lime Green */
            --alert-color: #FFFF33; /* Neon Yellow */
            --text-color: #FFFFFF; /* Crisp White */
            --heading-text-color: #F5F5F5; /* Adjusted for contrast */
        }

        h1, h2, h3 {
            color: var(--primary-color);
        }

        .stButton > button {
            background: var(--secondary-color);
            color: var(--text-color);
            border: none;
            padding: 1em 2em;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background-color: var(--accent-color); /* Fiery Red */
        }

        .stWarning {
            color: var(--alert-color); /* Neon Yellow */
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.header("Analysis Settings")

# Sport Selector
sport = st.selectbox("Select Sport", options=["NFL", "NBA"])

# Select Season Year(s)
season_year = st.number_input("Select Season Year", min_value=2000, max_value=2100, value=2023)

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xlsx"])

# Load data from file or fetch data based on user selection
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("Data Preview:")
    st.write(df.head())
else:
    fetch_data_option = st.checkbox("Fetch Data Using Selected Sport", value=False)
    if fetch_data_option:
        try:
            df = fetch_sport_data(sport, [season_year])
            st.write(f"Fetched and Cleaned {sport} Data for Season {season_year}:")
            st.write(df.head())
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

# Verify DataFrame existence
if 'df' not in locals():
    st.warning("Please upload a dataset or fetch data using the selected sport.")
    st.stop()

# Check if DataFrame is empty
if df.empty:
    st.warning("The uploaded dataset is empty. Please upload a valid dataset.")
    st.stop()

# Aggregate data if needed
aggregate_option = st.checkbox("Aggregate Data (e.g., for Play-by-Play)", value=False)
num_games = 1  # Default to 1 if no aggregation is applied

def aggregate_play_by_play_data(df, group_columns, agg_columns):
    agg_dict = {col: 'sum' for col in agg_columns}
    aggregated_df = df.groupby(group_columns).agg(agg_dict).reset_index()
    num_groups = aggregated_df.shape[0]
    return aggregated_df, num_groups

if aggregate_option:
    group_columns = st.multiselect("Select columns to group by (e.g., game_id, team)", df.columns)
    agg_columns = st.multiselect("Select columns to aggregate (e.g., rushing_yards, passing_yards)", df.select_dtypes(include='number').columns)

    if group_columns and agg_columns:
        df, num_games = aggregate_play_by_play_data(df, group_columns, agg_columns)
        st.write(f"Aggregated Data Preview (Number of unique groups: {num_games}):")
        st.write(df.head())

# Proceed with correlation and feature importance analysis
st.write("Proceeding with correlation and feature importance analysis...")
target_columns = st.multiselect("Select Target Column (e.g., fantasy_points)", df.select_dtypes(include='number').columns)

def calculate_feature_importance(df, features, target_column):
    df_clean = df.dropna(subset=features + [target_column])
    X = df_clean[features]
    y = df_clean[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    vif_data = calculate_vif(pd.DataFrame(X_scaled, columns=features))
    st.write("VIF (Variance Inflation Factor) to check multicollinearity:")
    st.write(vif_data)
    
    model = Ridge()
    model.fit(X_scaled, y)
    
    feature_importance = model.coef_
    model.feature_names_in_ = features  # Ensure this is a list
    
    return feature_importance, scaler, model

if target_columns:
    feature_columns = st.multiselect(
        "Select Feature Columns (e.g., passing_yards, rushing_yards)",
        [col for col in df.select_dtypes(include='number').columns if col not in target_columns]
    )

    if feature_columns:
        # Check for missing values in selected columns
        if df[feature_columns + target_columns].isnull().any().any():
            st.warning("Selected columns contain missing values. Please clean your data.")
            st.stop()

        correlation_result = calculate_correlation(df[feature_columns + target_columns], target_columns)
        st.write("Correlation Analysis Result:")
        st.write(correlation_result)

        st.write("Correlation Heatmap:")
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_result, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

        target_column = st.selectbox("Select a single Target Column for Feature Importance", target_columns)

        if target_column:
            final_feature_columns = [col for col in feature_columns if col != target_column]
            
            try:
                feature_importance, scaler, model = calculate_feature_importance(df, final_feature_columns, target_column)
            except ValueError as e:
                st.error(f"Error calculating feature importance: {e}")
                st.stop()

            # Display feature importance
            st.write(f"Feature Importance for predicting {target_column}:")
            importance_df = pd.DataFrame({
                'Feature': final_feature_columns,
                'Importance (Weight)': feature_importance
            }).sort_values(by='Importance (Weight)', ascending=False)

            st.write(importance_df)

            # Generate and display prediction formula
            formula_str = f"{target_column} = " + " + ".join([f"({weight:.3f}) * {feature}" for weight, feature in zip(importance_df['Importance (Weight)'], importance_df['Feature'])])
            st.write("Weighted Formula for Prediction:")
            st.write(formula_str)

            # Option to save model
            save_model_option = st.checkbox("Save trained model?", value=False)
            if save_model_option:
                model_name = st.text_input("Enter a name for your model:")
                if st.button("Save Model"):
                    if model_name:
                        try:
                            model_data = pickle.dumps((model, scaler))
                            metadata = f"Sport: {sport}, Season: {season_year}, Model trained on {target_column} with features {final_feature_columns}"
                            save_model(0, model_name, model_data, metadata)
                            st.success("Model saved successfully!")
                        except Exception as e:
                            st.error(f"Error saving model: {e}")
                    else:
                        st.warning("Please enter a valid model name.")

# Ensure the page runs independently as expected
