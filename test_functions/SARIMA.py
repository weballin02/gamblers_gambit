# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
from datetime import datetime
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

def nfl_predictions_sarima():
    st.title('NFL Team Points Prediction with SARIMA')

# Load and Preprocess Data
@st.cache_data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['schedule_date'] = pd.to_datetime(data['schedule_date'], errors='coerce')

    # Prepare home and away data
    home_df = data[['schedule_date', 'team_home', 'score_home']].copy()
    home_df.rename(columns={'team_home': 'team', 'score_home': 'score'}, inplace=True)

    away_df = data[['schedule_date', 'team_away', 'score_away']].copy()
    away_df.rename(columns={'team_away': 'team', 'score_away': 'score'}, inplace=True)

    # Combine both DataFrames
    team_data = pd.concat([home_df, away_df], ignore_index=True)
    team_data.dropna(subset=['score'], inplace=True)
    team_data['score'] = pd.to_numeric(team_data['score'], errors='coerce')
    team_data.set_index('schedule_date', inplace=True)
    team_data.sort_index(inplace=True)

    return team_data

# Load Data
file_path = 'data/nfl_data.csv'  # Update this path if necessary
team_data = load_and_preprocess_data(file_path)

# Get list of teams
teams_list = team_data['team'].unique()

# Train or Load SARIMA Models
@st.cache_resource
def get_team_models_sarima(team_data):
    model_dir = 'models/nfl_sarima/'  # Updated directory for SARIMA models
    os.makedirs(model_dir, exist_ok=True)

    team_models = {}
    teams_list = team_data['team'].unique()

    for team in teams_list:
        model_path = os.path.join(model_dir, f'{team}_sarima_model.pkl')

        team_scores = team_data[team_data['team'] == team]['score']
        team_scores = team_scores.asfreq('W')  # Ensure weekly frequency
        team_scores = team_scores.fillna(method='ffill')  # Handle missing data

        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            # Check if there are enough data points to train the model
            if len(team_scores) < 20:
                # Skip teams with insufficient data
                continue
            try:
                # Define SARIMA order (p, d, q) x (P, D, Q, S)
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 52)  # Assuming yearly seasonality

                model = SARIMAX(team_scores, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                joblib.dump(model_fit, model_path)
                model = model_fit
            except:
                continue

        team_models[team] = model

    return team_models

# Get Team SARIMA Models
team_models = get_team_models_sarima(team_data)

# Function to Predict Team Score using SARIMA
def predict_team_score_sarima(team, periods=1):
    model = team_models.get(team)
    if model:
        forecast = model.get

