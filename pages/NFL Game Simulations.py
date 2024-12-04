# Enhanced NFL Prediction Script with Advanced Features and Models

# =======================
# 1. Import Libraries
# =======================
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

warnings.filterwarnings("ignore")

# =======================
# 2. Initialization and Theme Configuration
# =======================
st.set_page_config(
    page_title="🏈 Advanced NFL Betting Predictions",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better visuals
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #FF4500;
        font-size: 48px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #2E8B57;
        font-size: 24px;
    }
    .metric-title {
        font-size: 18px;
        font-weight: bold;
    }
    .metric-value {
        font-size: 24px;
        color: #1E90FF;
    }
    </style>
    """, unsafe_allow_html=True)

# Theme Toggle Logic
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode
    if st.session_state.dark_mode:
        st.markdown("""
            <style>
            body {
                background-color: #2E2E2E;
                color: #FFFFFF;
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body {
                background-color: #FFFFFF;
                color: #000000;
            }
            </style>
            """, unsafe_allow_html=True)

st.sidebar.button("🌗 Toggle Theme", on_click=toggle_theme)

# =======================
# 3. Data Handling
# =======================
@st.cache_data(ttl=3600)
def load_nfl_data():
    current_year = datetime.now().year
    previous_years = [current_year - i for i in range(1, 6)]  # Last 5 years
    schedule = nfl.import_schedules([current_year] + previous_years)
    return schedule

@st.cache_data(ttl=3600)
def preprocess_data(schedule):
    # Prepare team data
    home_df = schedule[['gameday', 'home_team', 'home_score', 'away_team']].rename(
        columns={'home_team': 'team', 'home_score': 'score', 'away_team': 'opponent'}
    )
    home_df['is_home'] = 1

    away_df = schedule[['gameday', 'away_team', 'away_score', 'home_team']].rename(
        columns={'away_team': 'team', 'away_score': 'score', 'home_team': 'opponent'}
    )
    away_df['is_home'] = 0

    team_data = pd.concat([home_df, away_df], ignore_index=True)
    team_data.dropna(subset=['score'], inplace=True)
    team_data['gameday'] = pd.to_datetime(team_data['gameday'])
    team_data.sort_values(['team', 'gameday'], inplace=True)

    # Add rest days
    team_data['rest_days'] = team_data.groupby('team')['gameday'].diff().dt.days.fillna(7)

    # Add game number
    team_data['game_number'] = team_data.groupby('team').cumcount() + 1

    # Add rolling average of scores
    team_data['score_rolling_mean'] = team_data.groupby('team')['score'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # Add opponent's average score
    opponent_avg_score = team_data.groupby('team')['score'].mean().reset_index()
    opponent_avg_score.columns = ['opponent', 'opponent_avg_score']
    team_data = team_data.merge(opponent_avg_score, on='opponent', how='left')

    # Add exponential decay weights
    decay = 0.9
    team_data['decay_weight'] = team_data.groupby('team').cumcount().apply(lambda x: decay ** x)

    # Drop any remaining NaN values
    team_data.dropna(inplace=True)

    return team_data

schedule = load_nfl_data()
team_data = preprocess_data(schedule)

# =======================
# 4. Model Training
# =======================
def train_team_models(team_data):
    models = {}
    team_stats = {}
    for team in team_data['team'].unique():
        team_df = team_data[team_data['team'] == team]
        if len(team_df) < 15:
            continue  # Skip teams with insufficient data

        # Prepare features and target
        features = team_df[['game_number', 'is_home', 'rest_days', 'score_rolling_mean', 'opponent_avg_score']]
        target = team_df['score']

        # Handle missing values
        features.fillna(method='ffill', inplace=True)

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }

        # TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        # Initialize models
        gbr = GradientBoostingRegressor()
        rf = RandomForestRegressor()
        xgb = XGBRegressor(eval_metric='rmse', use_label_encoder=False)

        # Perform Grid Search
        gbr_grid = GridSearchCV(gbr, param_grid, cv=tscv)
        gbr_grid.fit(features, target)

        rf_grid = GridSearchCV(rf, {'n_estimators': [100, 200], 'max_depth': [5, 10]}, cv=tscv)
        rf_grid.fit(features, target)

        xgb_grid = GridSearchCV(xgb, param_grid, cv=tscv)
        xgb_grid.fit(features, target)

        # Store models
        models[team] = {
            'gbr': gbr_grid.best_estimator_,
            'rf': rf_grid.best_estimator_,
            'xgb': xgb_grid.best_estimator_,
            'features': features.columns
        }

        # Collect team stats
        team_stats[team] = {
            'avg_score': target.mean(),
            'std_dev': target.std(),
            'min_score': target.min(),
            'max_score': target.max()
        }

    return models, team_stats

models, team_stats = train_team_models(team_data)

# =======================
# 5. Prediction Logic
# =======================
def predict_team_score(team, models, team_stats, team_data):
    ensemble_prediction = None
    confidence_interval = None

    if team in models:
        team_df = team_data[team_data['team'] == team].iloc[-1]
        # Prepare features for the next game
        next_features = pd.DataFrame({
            'game_number': [team_df['game_number'] + 1],
            'is_home': [team_df['is_home']],  # Assume same as last game
            'rest_days': [7],  # Assume default rest days
            'score_rolling_mean': [team_df['score_rolling_mean']],
            'opponent_avg_score': [team_df['opponent_avg_score']]
        })

        # Get models
        model_dict = models[team]
        predictions = []
        for model_name in ['gbr', 'rf', 'xgb']:
            model = model_dict[model_name]
            pred = model.predict(next_features[model_dict['features']])[0]
            predictions.append(pred)

        # Ensemble prediction (average)
        ensemble_prediction = np.mean(predictions)

    if team in team_stats:
        avg = team_stats[team]['avg_score']
        std = team_stats[team]['std_dev']
        confidence_interval = (avg - 1.96 * std, avg + 1.96 * std)

    return ensemble_prediction, confidence_interval

# =======================
# 6. Fetch Upcoming Games
# =======================
@st.cache_data(ttl=3600)
def fetch_upcoming_games(schedule):
    upcoming_games = schedule[(schedule['home_score'].isna()) & (schedule['away_score'].isna())]
    upcoming_games['matchup'] = upcoming_games['home_team'] + " vs " + upcoming_games['away_team']
    upcoming_games['gameday'] = pd.to_datetime(upcoming_games['gameday'])
    upcoming_games.sort_values('gameday', inplace=True)
    return upcoming_games[['gameday', 'home_team', 'away_team', 'matchup']].reset_index(drop=True)

upcoming_games = fetch_upcoming_games(schedule)

# =======================
# 7. User Interface and Prediction Display
# =======================
# Title and Description
st.markdown('<div class="title">🏈 Advanced NFL Betting Predictions</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Get enhanced predictions with advanced modeling techniques for upcoming NFL games.</div>', unsafe_allow_html=True)

# Sidebar: Select an upcoming game
st.sidebar.header("📅 Upcoming Games")
if upcoming_games.empty:
    st.sidebar.warning("No upcoming games found.")
    st.stop()
selected_game = st.sidebar.selectbox(
    "Select a Game",
    upcoming_games['matchup']
)

# Get selected game details
selected_game_data = upcoming_games[upcoming_games['matchup'] == selected_game].iloc[0]
home_team = selected_game_data['home_team']
away_team = selected_game_data['away_team']

# Predictions for selected game
home_pred, home_ci = predict_team_score(home_team, models, team_stats, team_data)
away_pred, away_ci = predict_team_score(away_team, models, team_stats, team_data)

# Display Predictions
st.markdown("### 📊 Predictions")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### 🏠 **{home_team}**")
    if home_pred is not None and home_ci is not None:
        st.markdown(f"**Ensemble Prediction:** {home_pred:.2f}")
        st.markdown(f"**95% Confidence Interval:** {home_ci[0]:.2f} - {home_ci[1]:.2f}")
    else:
        st.warning("Insufficient data for predictions.")

with col2:
    st.markdown(f"#### 🚗 **{away_team}**")
    if away_pred is not None and away_ci is not None:
        st.markdown(f"**Ensemble Prediction:** {away_pred:.2f}")
        st.markdown(f"**95% Confidence Interval:** {away_ci[0]:.2f} - {away_ci[1]:.2f}")
    else:
        st.warning("Insufficient data for predictions.")

# Betting Insights
st.markdown("### 💡 Betting Insights")
with st.expander("View Betting Suggestions"):
    if home_pred is not None and away_pred is not None:
        suggested_winner = home_team if home_pred > away_pred else away_team
        margin_of_victory = abs(home_pred - away_pred)
        total_points = home_pred + away_pred
        over_under_threshold = team_data['score'].mean() * 2  # Dynamic threshold
        over_under = "Over" if total_points > over_under_threshold else "Under"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**🏆 Suggested Winner:** **{suggested_winner}**")

        with col2:
            st.markdown(f"**📈 Margin of Victory:** {margin_of_victory:.2f} pts")

        with col3:
            st.markdown(f"**🔢 Total Points:** {total_points:.2f}")

        st.markdown(f"**💰 Over/Under Suggestion:** **Take the {over_under} ({over_under_threshold:.2f})**")
    else:
        st.warning("Insufficient data to provide betting insights for this game.")

# Visual Comparison
st.markdown("### 📊 Predicted Scores Comparison")
scores_df = pd.DataFrame({
    "Team": [home_team, away_team],
    "Predicted Score": [
        home_pred if home_pred is not None else 0,
        away_pred if away_pred is not None else 0
    ]
})

# Customize the bar chart with Seaborn
fig, ax = plt.subplots(figsize=(8, 6))
sns.set_style("whitegrid")
sns.barplot(x="Team", y="Predicted Score", data=scores_df, palette="viridis", ax=ax)
ax.set_title("Predicted Scores")
ax.set_ylabel("Score")
ax.set_ylim(0, max(scores_df["Predicted Score"].max(), 50) + 10 if scores_df["Predicted Score"].max() > 0 else 50)

# Annotate bars with values
for index, row in scores_df.iterrows():
    ax.text(index, row["Predicted Score"] + 0.5, f'{row["Predicted Score"]:.2f}', color='black', ha="center")

st.pyplot(fig)

# Feature Importance Visualization
st.markdown("### 🔍 Feature Importance")
with st.expander("View Feature Importance for Home Team"):
    if home_team in models:
        model_dict = models[home_team]
        model = model_dict['xgb']  # Using XGBoost for feature importance
        explainer = shap.TreeExplainer(model)
        team_df = team_data[team_data['team'] == home_team]
        team_features = team_df[model_dict['features']].iloc[-1:]
        shap_values = explainer.shap_values(team_features)
        shap.initjs()
        plt.title(f"Feature Importance for {home_team}")
        shap.summary_plot(shap_values, team_features, plot_type="bar")
        st.pyplot(bbox_inches='tight')
    else:
        st.warning("Insufficient data for feature importance visualization.")

with st.expander("View Feature Importance for Away Team"):
    if away_team in models:
        model_dict = models[away_team]
        model = model_dict['xgb']
        explainer = shap.TreeExplainer(model)
        team_df = team_data[team_data['team'] == away_team]
        team_features = team_df[model_dict['features']].iloc[-1:]
        shap_values = explainer.shap_values(team_features)
        shap.initjs()
        plt.title(f"Feature Importance for {away_team}")
        shap.summary_plot(shap_values, team_features, plot_type="bar")
        st.pyplot(bbox_inches='tight')
    else:
        st.warning("Insufficient data for feature importance visualization.")

# Additional Team Statistics (Optional)
st.markdown("### 📋 Team Statistics")
with st.expander("View Team Performance Stats"):
    stats_df = pd.DataFrame(team_stats).T
    stats_df = stats_df.rename_axis("Team").reset_index()
    st.dataframe(stats_df.style.highlight_max(axis=0), use_container_width=True)
