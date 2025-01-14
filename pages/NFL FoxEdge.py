# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
from pmdarima import auto_arima
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from scipy import stats
from scipy.special import expit  # Sigmoid function

warnings.filterwarnings('ignore')

# Streamlit App Configuration
st.set_page_config(
    page_title="🏈 NFL FoxEdge",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State for Theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Function to Toggle Theme
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Theme Toggle Button
st.button("🌗 Toggle Theme", on_click=toggle_theme)

# Apply Theme Based on Dark Mode
if st.session_state.dark_mode:
    primary_bg = "#121212"
    secondary_bg = "#1E1E1E"
    primary_text = "#FFFFFF"
    secondary_text = "#B0B0B0"
    accent_color = "#BB86FC"  # Purple
    highlight_color = "#03DAC6"  # Teal
    chart_color = "#BB86FC"
    chart_template = "plotly_dark"
else:
    primary_bg = "#FFFFFF"
    secondary_bg = "#F0F0F0"
    primary_text = "#000000"
    secondary_text = "#4F4F4F"
    accent_color = "#6200EE"
    highlight_color = "#03DAC6"
    chart_color = "#6200EE"
    chart_template = "plotly_white"

# Custom CSS Styling
st.markdown(f'''
    <style>
        /* Global Styles */
        body, html {{
            background-color: {primary_bg};
            color: {primary_text};
            font-family: 'Open Sans', sans-serif;
        }}

        /* Hide Streamlit Branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Header */
        .header-title {{
            font-family: 'Raleway', sans-serif;
            font-size: 3em;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.5em;
            background: linear-gradient(90deg, {highlight_color}, {accent_color});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header-description {{
            text-align: center;
            color: {secondary_text};
            font-size: 1.2em;
            margin-bottom: 2em;
        }}

        /* Buttons */
        .stButton > button {{
            background: linear-gradient(90deg, {highlight_color}, {accent_color});
            color: {primary_text};
            border: none;
            padding: 0.8em 2em;
            border-radius: 30px;
            font-size: 1em;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.3s ease, background 0.3s ease;
        }}

        .stButton > button:hover {{
            transform: translateY(-5px);
            background: linear-gradient(90deg, {accent_color}, {highlight_color});
        }}

        /* Data Section */
        .data-section {{
            padding: 2em 1em;
            background-color: {secondary_bg};
            border-radius: 15px;
            margin-bottom: 2em;
        }}

        .data-section h2 {{
            font-size: 2.5em;
            margin-bottom: 0.5em;
            color: {accent_color};
        }}

        .data-section p {{
            font-size: 1.2em;
            color: {secondary_text};
            margin-bottom: 2em;
        }}

        /* Team Trends Styling */
        .team-trends {{
            display: flex;
            flex-wrap: wrap;
            gap: 2em;
            justify-content: space-around;
            margin-top: 2em;
        }}

        .team-card {{
            background-color: {secondary_bg};
            border-radius: 15px;
            padding: 1.5em;
            width: calc(33% - 2em);
            min-width: 300px;
            max-width: 400px;
            text-align: center;
            transition: transform 0.3s ease;
        }}

        .team-card:hover {{
            transform: translateY(-5px);
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 2em 0;
            color: {secondary_text};
            font-size: 0.9em;
        }}

        .footer a {{
            color: {accent_color};
            text-decoration: none;
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            .header-title {{
                font-size: 2em;
            }}

            .header-description {{
                font-size: 1em;
            }}

            .team-trends {{
                flex-direction: column;
                align-items: center;
            }}

            .team-card {{
                width: 90%;
            }}
        }}
    </style>
''', unsafe_allow_html=True)

# Header Section
st.markdown(f'''
    <h1 class="header-title">NFL FoxEdge</h1>
    <p class="header-description">Matchup Predictions and Betting Recommendations</p>
''', unsafe_allow_html=True)

st.markdown("""
Welcome to the **NFL FoxEdge** app! Here, you'll find game predictions, betting leans, and in-depth analysis to help you make informed betting decisions. Whether you're a casual bettor or a seasoned professional, our insights are tailored to provide value to all.
""")

# Team Abbreviation to Full Name Mapping
team_abbrev_mapping = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs',
    'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams',
    'LV': 'Las Vegas Raiders',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NO': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SEA': 'Seattle Seahawks',
    'SF': 'San Francisco 49ers',
    'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders',
}

# Invert the mapping for reverse lookup
full_name_to_abbrev = {v: k for k, v in team_abbrev_mapping.items()}

# Get list of team abbreviations
team_abbreviations = list(team_abbrev_mapping.keys())
team_name_mapping = team_abbrev_mapping.copy()

# Fetch and Preprocess Data
@st.cache_data
def fetch_and_preprocess_data(season_years):
    """
    Fetch data for the specified seasons and preprocess it.
    """
    all_team_data = []
    failed_teams = []
    for year in season_years:
        try:
            schedule = nfl.import_schedules([year])
            schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
            schedule['Week_Number'] = schedule['week']

            # Prepare home and away data
            home_df = schedule[['gameday', 'home_team', 'home_score', 'away_score', 'Week_Number']].copy()
            home_df.rename(columns={
                'gameday': 'GAME_DATE',
                'home_team': 'TEAM_ABBREV',
                'home_score': 'PTS',
                'away_score': 'OPP_PTS'
            }, inplace=True)
            home_df['Home_Away'] = 'Home'

            away_df = schedule[['gameday', 'away_team', 'away_score', 'home_score', 'Week_Number']].copy()
            away_df.rename(columns={
                'gameday': 'GAME_DATE',
                'away_team': 'TEAM_ABBREV',
                'away_score': 'PTS',
                'home_score': 'OPP_PTS'
            }, inplace=True)
            away_df['Home_Away'] = 'Away'

            # Combine both DataFrames
            season_data = pd.concat([home_df, away_df], ignore_index=True)
            season_data.dropna(subset=['PTS'], inplace=True)
            season_data['PTS'] = pd.to_numeric(season_data['PTS'], errors='coerce')
            season_data['OPP_PTS'] = pd.to_numeric(season_data['OPP_PTS'], errors='coerce')
            season_data['SEASON'] = year

            # Calculate Points Allowed (Defensive Stat)
            season_data['Points_Allowed'] = season_data['OPP_PTS']

            # Calculate Days Since Last Game (Rest Days)
            season_data.sort_values(['TEAM_ABBREV', 'GAME_DATE'], inplace=True)
            season_data['Days_Since_Last_Game'] = season_data.groupby('TEAM_ABBREV')['GAME_DATE'].diff().dt.days.fillna(7)

            all_team_data.append(season_data)
        except Exception as e:
            st.warning(f"Error fetching data for season {year}: {e}")
            failed_teams.append(year)
            continue
    if all_team_data:
        data = pd.concat(all_team_data, ignore_index=True)
        data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])
        return data, failed_teams
    else:
        return pd.DataFrame(), failed_teams

# Fetch Data
current_year = datetime.now().year
season_years = [current_year - 2, current_year - 1, current_year]
data, failed_teams = fetch_and_preprocess_data(season_years)

# Handle Outliers
def remove_outliers(df):
    df = df.copy()
    numeric_cols = ['PTS', 'OPP_PTS', 'Points_Allowed']
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        df = df[z_scores < 3]
    return df

data = remove_outliers(data)

# Aggregate Points by Team and Date
def aggregate_points_by_team(data):
    team_data = data.groupby(['GAME_DATE', 'TEAM_ABBREV']).agg({
        'PTS': 'sum',
        'OPP_PTS': 'sum',
        'Home_Away': 'first',
        'Week_Number': 'first',
        'Points_Allowed': 'sum',
        'Days_Since_Last_Game': 'first',
        'SEASON': 'first'
    }).reset_index()
    team_data.set_index('GAME_DATE', inplace=True)
    return team_data

team_data = aggregate_points_by_team(data)

# Add Rolling Statistics
def add_rolling_stats(team_data):
    team_data = team_data.copy()
    team_data.sort_values(['TEAM_ABBREV', 'GAME_DATE'], inplace=True)
    team_data['Rolling_Avg_PTS'] = team_data.groupby('TEAM_ABBREV')['PTS'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    team_data['Rolling_Std_PTS'] = team_data.groupby('TEAM_ABBREV')['PTS'].transform(lambda x: x.rolling(window=3, min_periods=1).std())
    team_data['Rolling_Avg_Points_Allowed'] = team_data.groupby('TEAM_ABBREV')['Points_Allowed'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    return team_data

team_data = add_rolling_stats(team_data)

# Standardize Features (Exclude 'PTS')
def standardize_features(team_data):
    scaler = StandardScaler()
    features_to_scale = ['OPP_PTS', 'Points_Allowed', 'Rolling_Avg_PTS', 'Rolling_Std_PTS', 'Rolling_Avg_Points_Allowed']
    team_data[features_to_scale] = scaler.fit_transform(team_data[features_to_scale])
    return team_data, scaler

team_data, feature_scaler = standardize_features(team_data)

# Train or Load Models with Cross-Validation and Fine-Tuning
@st.cache_resource
def train_models(team_data):
    model_dir = 'models/nfl'
    os.makedirs(model_dir, exist_ok=True)
    team_models = {}
    team_mae = {}
    teams = team_data['TEAM_ABBREV'].unique()

    for team in teams:
        model_path = os.path.join(model_dir, f"{team}_arima_model.pkl")
        team_df = team_data[team_data['TEAM_ABBREV'] == team]
        team_points = team_df['PTS']
        team_points.index = team_df.index  # Ensure index is date for time series

        if len(team_points) < 10:
            st.warning(f"Not enough data to train model for {team}.")
            continue

        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=3)
        errors = []
        for train_index, test_index in tscv.split(team_points):
            train, test = team_points.iloc[train_index], team_points.iloc[test_index]

            # ARIMA Model
            try:
                arima_model = auto_arima(
                    train,
                    start_p=0, max_p=3,
                    start_q=0, max_q=3,
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                arima_forecast = arima_model.predict(n_periods=len(test))
                mae = mean_absolute_error(test, arima_forecast)
                errors.append(mae)
            except Exception as e:
                st.warning(f"Error training ARIMA model for {team}: {e}")
                continue

        # Average MAE
        avg_mae = np.mean(errors) if errors else None
        if avg_mae is not None:
            st.write(f"Average MAE for {team}: {avg_mae:.2f}")
            team_mae[team] = avg_mae
        else:
            st.warning(f"No MAE calculated for {team} due to training errors.")

        # Retrain on full data
        try:
            final_arima_model = auto_arima(
                team_points,
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            # Save the model
            joblib.dump(final_arima_model, model_path)
            team_models[team] = final_arima_model
        except Exception as e:
            st.warning(f"Error retraining ARIMA model for {team}: {e}")
            continue

    return team_models, team_mae

team_models, team_mae_dict = train_models(team_data)

# Compute Team Forecasts
def compute_team_forecasts(team_models, team_data):
    team_forecasts = {}
    forecast_periods = 1  # Predict next game
    for team_abbrev, model in team_models.items():
        if team_abbrev not in team_data['TEAM_ABBREV'].values:
            st.warning(f"No data available for team: {team_abbrev}")
            continue
        team_df = team_data[team_data['TEAM_ABBREV'] == team_abbrev]
        if team_df.empty:
            continue
        last_date = team_df.index.max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=forecast_periods, freq='7D')
        forecast = model.predict(n_periods=forecast_periods)
        predictions = pd.DataFrame({'Date': future_dates, 'Predicted_PTS': forecast, 'Team': team_abbrev})
        team_forecasts[team_abbrev] = predictions
    return pd.concat(team_forecasts.values(), ignore_index=True) if team_forecasts else pd.DataFrame(columns=['Date', 'Predicted_PTS', 'Team'])

team_forecasts = compute_team_forecasts(team_models, team_data)

# Fetch upcoming games based on the current day of the week
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    schedule = nfl.import_schedules([current_year])
    schedule['game_datetime'] = pd.to_datetime(schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str), errors='coerce', utc=True)
    now = datetime.now(pytz.UTC)
    weekday = now.weekday()

    target_days = {3: [3, 6, 0], 6: [6, 0, 3], 0: [0, 3, 6]}.get(weekday, [3, 6, 0])
    upcoming_game_dates = [now + timedelta(days=(d - weekday + 7) % 7) for d in target_days]

    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'].dt.date.isin([date.date() for date in upcoming_game_dates]))
    ].sort_values('game_datetime')

    return upcoming_games[['game_datetime', 'home_team', 'away_team']]

upcoming_games = fetch_upcoming_games()

# Fetch Current Season Stats
def fetch_current_season_stats():
    schedule = nfl.import_schedules([current_year])
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    home_df = schedule[['gameday', 'home_team', 'home_score', 'away_score']].copy().rename(columns={'gameday': 'GAME_DATE', 'home_team': 'TEAM_ABBREV', 'home_score': 'PTS', 'away_score': 'OPP_PTS'})
    home_df['Home_Away'] = 'Home'
    away_df = schedule[['gameday', 'away_team', 'away_score', 'home_score']].copy().rename(columns={'gameday': 'GAME_DATE', 'away_team': 'TEAM_ABBREV', 'away_score': 'PTS', 'home_score': 'OPP_PTS'})
    away_df['Home_Away'] = 'Away'
    current_season_data = pd.concat([home_df, away_df], ignore_index=True)
    current_season_data.dropna(subset=['PTS'], inplace=True)
    current_season_data.set_index('GAME_DATE', inplace=True)
    current_season_data.sort_index(inplace=True)
    # Calculate team stats
    team_stats = current_season_data.groupby('TEAM_ABBREV').apply(
        lambda x: pd.Series({
            'avg_score': x['PTS'].mean(),
            'max_score': x['PTS'].max(),
            'std_dev': x['PTS'].std(),
            'games_played': x['PTS'].count(),
            'recent_form': x['PTS'].tail(5).mean() if len(x) >= 5 else x['PTS'].mean(),
            'avg_points_allowed': x['OPP_PTS'].mean()
        })
    ).to_dict(orient='index')
    return team_stats

current_season_stats = fetch_current_season_stats()

# Predict Matchups with Enhanced Features
def predict_game_outcome(home_team, away_team, team_forecasts, current_season_stats, team_data, model_mae_dict):
    home_forecast = team_forecasts[team_forecasts['Team'] == home_team]['Predicted_PTS'].mean()
    away_forecast = team_forecasts[team_forecasts['Team'] == away_team]['Predicted_PTS'].mean()
    home_stats = current_season_stats.get(home_team, {})
    away_stats = current_season_stats.get(away_team, {})
    
    if home_stats and away_stats and not np.isnan(home_forecast) and not np.isnan(away_forecast):
        # Strength of Schedule Adjustment (Simplified Example)
        home_opponent_avg = team_data[team_data['TEAM_ABBREV'] == away_team]['PTS'].mean()
        away_opponent_avg = team_data[team_data['TEAM_ABBREV'] == home_team]['PTS'].mean()
        home_strength_adjustment = home_stats['avg_score'] - away_opponent_avg
        away_strength_adjustment = away_stats['avg_score'] - home_opponent_avg

        # Combine all features for rating
        home_team_rating = (
            0.25 * home_stats['avg_score'] +
            0.15 * home_stats['max_score'] +
            0.2 * home_stats['recent_form'] +
            0.15 * home_forecast +
            0.1 * home_strength_adjustment -
            0.1 * home_stats['std_dev'] +
            0.05 * (1 if home_stats['games_played'] > away_stats['games_played'] else 0)
        )

        away_team_rating = (
            0.25 * away_stats['avg_score'] +
            0.15 * away_stats['max_score'] +
            0.2 * away_stats['recent_form'] +
            0.15 * away_forecast +
            0.1 * away_strength_adjustment -
            0.1 * away_stats['std_dev'] +
            0.05 * (1 if away_stats['games_played'] > home_stats['games_played'] else 0)
        )

        rating_diff = home_team_rating - away_team_rating

        # Retrieve model's average MAE for both teams
        home_mae = model_mae_dict.get(home_team, 2.0)  # Default MAE=2.0 if not available
        away_mae = model_mae_dict.get(away_team, 2.0)  # Default MAE=2.0 if not available

        # Calculate average MAE
        avg_mae = (home_mae + away_mae) / 2

        # Normalize rating_diff between -1 and 1
        max_rating = max(abs(home_team_rating), abs(away_team_rating))
        normalized_rating_diff = rating_diff / max_rating if max_rating != 0 else 0

        # Apply sigmoid function to map to (0,1)
        sigmoid_confidence = expit(normalized_rating_diff * 6)  # Multiplier adjusts the steepness

        # Adjust confidence based on model's average MAE
        mae_adjustment = 1.2 - (avg_mae / 10)  # Example scaling; adjust as needed
        mae_adjustment = np.clip(mae_adjustment, 0.8, 1.2)  # Clamp between 0.8 and 1.2

        # Final confidence scaled to 0-100%
        confidence = sigmoid_confidence * 100 * mae_adjustment

        # Clamp the confidence between 0 and 100
        confidence = min(100, max(0, confidence))

        # Determine predicted winner based on forecasted scores
        if home_forecast > away_forecast:
            predicted_winner = home_team
            predicted_score_diff = home_forecast - away_forecast
        else:
            predicted_winner = away_team
            predicted_score_diff = away_forecast - home_forecast

        return predicted_winner, predicted_score_diff, confidence, home_forecast, away_forecast
    else:
        return "Unavailable", "N/A", "N/A", None, None

# Main logic for predicting upcoming games
if st.button("Predict Upcoming Games"):
    if data.empty:
        st.error("No data available for analysis.")
    else:
        results = []
        if not upcoming_games.empty:
            for _, game in upcoming_games.iterrows():
                home_team = game["home_team"]
                away_team = game["away_team"]
                game_datetime = game["game_datetime"]

                result = predict_game_outcome(home_team, away_team, team_forecasts, current_season_stats, team_data, team_mae_dict)

                winner, score_diff, confidence, home_forecast, away_forecast = result

                if pd.isna(score_diff) or pd.isna(confidence) or winner == "Unavailable":
                    st.warning(f"Data unavailable for matchup: {team_name_mapping.get(home_team, home_team)} vs {team_name_mapping.get(away_team, away_team)}")
                    continue

                # Display Matchup Details
                st.markdown(f"<h2 style='color:{accent_color};'>{team_name_mapping.get(home_team, home_team)} vs {team_name_mapping.get(away_team, away_team)}</h2>", unsafe_allow_html=True)
                st.markdown(f"- **Game Date:** {game_datetime.strftime('%Y-%m-%d %H:%M %Z')}")
                st.markdown(f"- **Predicted Winner:** **{team_name_mapping.get(winner, 'Unavailable')}**")
                st.markdown(f"- **Predicted Final Scores:** {team_name_mapping.get(home_team, home_team)} **{home_forecast:.2f}** - {team_name_mapping.get(away_team, away_team)} **{away_forecast:.2f}**")
                st.markdown(f"- **Score Difference:** **{score_diff:.2f}**")
                st.markdown(f"- **Confidence Level:** **{confidence:.2f}%**")

                # Betting Recommendations
                spread_lean = f"Lean {team_name_mapping.get(winner, winner)} -{int(score_diff)}"
                total_points = home_forecast + away_forecast
                total_recommendation = f"Lean Over if the line is < {total_points:.2f}"
                st.markdown(f"- **Spread Bet:** {spread_lean}")
                st.markdown(f"- **Total Points Bet:** {total_recommendation}")
                if confidence > 75:
                    st.success(f"High Confidence: {confidence:.2f}%")
                elif confidence > 50:
                    st.warning(f"Moderate Confidence: {confidence:.2f}%")
                else:
                    st.error(f"Low Confidence: {confidence:.2f}%")

                # Radar Chart
                categories = ["Average Score", "Max Score", "Consistency", "Recent Form", "Predicted PTS"]
                home_stats = current_season_stats.get(home_team, {})
                away_stats = current_season_stats.get(away_team, {})
                home_stats_values = [
                    home_stats.get('avg_score', 0),
                    home_stats.get('max_score', 0),
                    1 / home_stats.get('std_dev', 1) if home_stats.get('std_dev', 1) != 0 else 1,  # Consistency
                    home_stats.get('recent_form', 0),
                    home_forecast
                ]
                away_stats_values = [
                    away_stats.get('avg_score', 0),
                    away_stats.get('max_score', 0),
                    1 / away_stats.get('std_dev', 1) if away_stats.get('std_dev', 1) != 0 else 1,  # Consistency
                    away_stats.get('recent_form', 0),
                    away_forecast
                ]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=home_stats_values, theta=categories, fill='toself', name=f"{team_name_mapping.get(home_team, home_team)}"))
                fig.add_trace(go.Scatterpolar(r=away_stats_values, theta=categories, fill='toself', name=f"{team_name_mapping.get(away_team, away_team)}"))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=True,
                    title=f"Team Comparison: {team_name_mapping.get(home_team, home_team)} vs {team_name_mapping.get(away_team, away_team)}",
                    template=chart_template
                )
                st.plotly_chart(fig)

                # Append to results
                results.append({
                    'Home_Team': home_team,
                    'Away_Team': away_team,
                    'Predicted Winner': winner,
                    'Spread Lean': spread_lean,
                    'Total Lean': total_recommendation,
                    'Confidence': confidence
                })

            # Visualizations
            if results:
                # Confidence Heatmap
                st.header("Confidence Heatmap for Upcoming Matchups")
                confidence_data = pd.DataFrame(results)
                heatmap_data = confidence_data.pivot("Home_Team", "Away_Team", "Confidence")
                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data, annot=True, cmap="RdYlGn", cbar_kws={'label': 'Confidence Level'}, fmt=".1f")
                plt.title("Confidence Heatmap for Upcoming Matchups")
                plt.xlabel("Away Team")
                plt.ylabel("Home Team")
                st.pyplot(plt)

                # Betting Summary
                st.header("📊 Recommended Bets Summary")
                summary_df = pd.DataFrame(results)
                st.dataframe(summary_df[['Home_Team', 'Away_Team', 'Predicted Winner', 'Spread Lean', 'Total Lean', 'Confidence']].style.highlight_max(subset=['Confidence'], color='lightgreen'))

                # Confidence Breakdown
                st.header("Confidence Level Breakdown")
                confidence_counts = pd.Series([result['Confidence'] for result in results]).value_counts(bins=[0,50,75,100], sort=False)
                labels = ['Low (0-50%)', 'Moderate (50-75%)', 'High (75-100%)']
                plt.figure(figsize=(8, 6))
                plt.pie(confidence_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                plt.title("Confidence Level Breakdown")
                st.pyplot(plt)

                # Best Bets
                st.header("🔥 Top Betting Opportunities")
                best_bets = [result for result in results if result['Confidence'] > 80]
                if best_bets:
                    for bet in best_bets:
                        opponent_team = bet['Away_Team'] if bet['Predicted Winner'] == bet['Home_Team'] else bet['Home_Team']
                        st.markdown(f"**{team_name_mapping.get(bet['Predicted Winner'], bet['Predicted Winner'])}** over **{team_name_mapping.get(opponent_team, opponent_team)}**")
                        st.markdown(f"- **Spread Bet:** {bet['Spread Lean']}")
                        st.markdown(f"- **Total Points Bet:** {bet['Total Lean']}")
                        st.markdown(f"- **Confidence Level:** **{bet['Confidence']:.2f}%**")
                else:
                    st.write("No high confidence bets found.")

                # Additional Insights
                st.header("Detailed Analysis for Professional Bettors")
                st.write("Here are the in-depth stats and trends for each matchup.")
                for result in results:
                    home_team = result['Home_Team']
                    away_team = result['Away_Team']
                    home_stats = current_season_stats.get(home_team, {})
                    away_stats = current_season_stats.get(away_team, {})
                    home_team_name = team_name_mapping.get(home_team, home_team)
                    away_team_name = team_name_mapping.get(away_team, away_team)
                    st.subheader(f"{home_team_name} vs {away_team_name}")
                    st.write("**Home Team Stats:**")
                    st.write(home_stats)
                    st.write("**Away Team Stats:**")
                    st.write(away_stats)
                    # Plot team performance trends
                    team_data_home = data[data['TEAM_ABBREV'] == home_team]
                    team_data_away = data[data['TEAM_ABBREV'] == away_team]
                    st.write(f"**{home_team_name} Performance Trends:**")
                    plt.figure(figsize=(10, 5))
                    plt.plot(team_data_home['GAME_DATE'], team_data_home['PTS'], label='Points Scored', marker='o', color='blue')
                    plt.title(f"{home_team_name} Performance Trends")
                    plt.xlabel("Date")
                    plt.ylabel("Points")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt)
                    st.write(f"**{away_team_name} Performance Trends:**")
                    plt.figure(figsize=(10, 5))
                    plt.plot(team_data_away['GAME_DATE'], team_data_away['PTS'], label='Points Scored', marker='o', color='red')
                    plt.title(f"{away_team_name} Performance Trends")
                    plt.xlabel("Date")
                    plt.ylabel("Points")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt)

            else:
                st.warning("No matchups to display.")

        if failed_teams:
            st.warning(f"Data could not be fetched for the following teams: {', '.join(failed_teams)}")

    # User Education and Responsible Gambling Reminder
    st.markdown(f"""
    **Understanding the Metrics:**
    - **Predicted Winner:** The team our model predicts will win the game.
    - **Spread Bet:** Our suggested bet against the point spread.
    - **Total Points Bet:** Our recommendation for the over/under total points.
    - **Confidence Level:** How confident we are in our prediction (0-100%).
    
    **Please Gamble Responsibly:**
    - Set limits and stick to them.
    - Don't chase losses.
    - If you need help, visit [Gambling Support](https://www.example.com/gambling-support).
    
    **New to sports betting?**
    - [Understanding Point Spreads](https://www.example.com/point-spreads)
    - [Guide to Over/Under Betting](https://www.example.com/over-under)
    - [Betting Strategies for Beginners](https://www.example.com/betting-strategies)
    
    **Data last updated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)

    # Footer
    st.markdown(f'''
        <div class="footer">
            &copy; {datetime.now().year} **NFL FoxEdge**. All rights reserved.
        </div>
    ''', unsafe_allow_html=True)
