


# Enhanced NFL Prediction Script with Betting Insights and Improved Visuals

# =======================
# 1. Import Libraries
# =======================
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.ensemble import GradientBoostingRegressor
from pmdarima import auto_arima
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# =======================
# 2. Initialization and Theme Configuration
# =======================
st.set_page_config(
    page_title="🏈 Enhanced NFL Betting Predictions",
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
    previous_years = [current_year - 1, current_year - 2]
    schedule = nfl.import_schedules([current_year] + previous_years)
    return schedule

@st.cache_data(ttl=3600)
def preprocess_data(schedule):
    home_df = schedule[['gameday', 'home_team', 'home_score']].rename(
        columns={'home_team': 'team', 'home_score': 'score'}
    )
    away_df = schedule[['gameday', 'away_team', 'away_score']].rename(
        columns={'away_team': 'team', 'away_score': 'score'}
    )
    team_data = pd.concat([home_df, away_df], ignore_index=True)
    team_data.dropna(subset=['score'], inplace=True)
    team_data.sort_values('gameday', inplace=True)  # Ensure data is in chronological order
    return team_data

schedule = load_nfl_data()
team_data = preprocess_data(schedule)

# =======================
# 4. Model Training
# =======================
def train_team_models(team_data):
    models = {}
    arima_models = {}
    team_stats = {}
    for team in team_data['team'].unique():
        team_scores = team_data[team_data['team'] == team]['score'].reset_index(drop=True)
        team_stats[team] = {
            'avg_score': team_scores.mean(),
            'std_dev': team_scores.std(),
            'min_score': team_scores.min(),
            'max_score': team_scores.max()
        }
        if len(team_scores) > 10:
            features = np.arange(len(team_scores)).reshape(-1, 1)
            gbr_model = GradientBoostingRegressor().fit(features, team_scores)
            models[team] = gbr_model
        if len(team_scores) > 5:
            arima_model = auto_arima(team_scores, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
            arima_models[team] = arima_model
    return models, arima_models, team_stats

gbr_models, arima_models, team_stats = train_team_models(team_data)

# =======================
# 5. Prediction Logic
# =======================
def predict_team_score(team, models, arima_models, team_stats, team_data):
    gbr_prediction = None
    arima_prediction = None
    confidence_interval = None
    # Get the team's scores
    team_scores = team_data[team_data['team'] == team]['score'].reset_index(drop=True)
    if team in models:
        gbr_model = models[team]
        # The next feature is the index for the next game
        next_feature = np.array([[len(team_scores)]])
        gbr_prediction = gbr_model.predict(next_feature)[0]
    if team in arima_models:
        arima_model = arima_models[team]
        arima_pred = arima_model.predict(n_periods=1)
        if isinstance(arima_pred, pd.Series):
            arima_prediction = arima_pred.iloc[0]
        else:
            arima_prediction = arima_pred[0]
    if team in team_stats:
        avg = team_stats[team]['avg_score']
        std = team_stats[team]['std_dev']
        confidence_interval = (avg - 1.96 * std, avg + 1.96 * std)
    return gbr_prediction, arima_prediction, confidence_interval

# =======================
# 6. Fetch Upcoming Games
# =======================
@st.cache_data(ttl=3600)
def fetch_upcoming_games(schedule):
    upcoming_games = schedule[(schedule['home_score'].isna()) & (schedule['away_score'].isna())]
    upcoming_games['matchup'] = upcoming_games['home_team'] + " vs " + upcoming_games['away_team']
    return upcoming_games[['gameday', 'home_team', 'away_team', 'matchup']].reset_index(drop=True)

upcoming_games = fetch_upcoming_games(schedule)

# =======================
# 7. User Interface and Prediction Display
# =======================
# Title and Description
st.markdown('<div class="title">🏈 Enhanced NFL Betting Predictions</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Get predictions and actionable betting insights for upcoming NFL games.</div>', unsafe_allow_html=True)

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
home_gbr, home_arima, home_ci = predict_team_score(home_team, gbr_models, arima_models, team_stats, team_data)
away_gbr, away_arima, away_ci = predict_team_score(away_team, gbr_models, arima_models, team_stats, team_data)

# Display Predictions
st.markdown("### 📊 Predictions")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### 🟢 **{home_team}**")
    if home_gbr is not None and home_arima is not None and home_ci is not None:
        st.markdown(f"**Gradient Boosting Prediction:** {home_gbr:.2f}")
        st.markdown(f"**ARIMA Prediction:** {home_arima:.2f}")
        st.markdown(f"**95% Confidence Interval:** {home_ci[0]:.2f} - {home_ci[1]:.2f}")
    else:
        st.warning("Insufficient data for predictions.")

with col2:
    st.markdown(f"#### 🟠 **{away_team}**")
    if away_gbr is not None and away_arima is not None and away_ci is not None:
        st.markdown(f"**Gradient Boosting Prediction:** {away_gbr:.2f}")
        st.markdown(f"**ARIMA Prediction:** {away_arima:.2f}")
        st.markdown(f"**95% Confidence Interval:** {away_ci[0]:.2f} - {away_ci[1]:.2f}")
    else:
        st.warning("Insufficient data for predictions.")
        
# Filter data for the selected home team
team_df = team_data[team_data['team'] == home_team]
if not team_df.empty:
    # Extract the latest features for the team
    team_features = team_df[model_dict['features']].iloc[-1:]

    # Compute SHAP values for the team
    shap_values = explainer.shap_values(team_features)

    # Display SHAP summary plot as a bar chart using Matplotlib
    st.markdown(f"### 🔍 Feature Importance for **{home_team}**")
    plt.figure(figsize=(10, 5))
    plt.title(f"Feature Importance for {home_team}")
    shap.summary_plot(shap_values, team_features, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
else:
    st.warning(f"No data available for {home_team}. Cannot generate SHAP visualizations.")

# Betting Insights
st.markdown("### 💡 Betting Insights")
with st.expander("View Betting Suggestions"):
    if home_gbr is not None and away_gbr is not None:
        suggested_winner = home_team if home_gbr > away_gbr else away_team
        margin_of_victory = abs(home_gbr - away_gbr)
        total_points = home_gbr + away_gbr
        over_under = "Over" if total_points > 45 else "Under"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**🏆 Suggested Winner:** **{suggested_winner}**")

        with col2:
            st.markdown(f"**📈 Margin of Victory:** {margin_of_victory:.2f} pts")

        with col3:
            st.markdown(f"**🔢 Total Points:** {total_points:.2f}")

        st.markdown(f"**💰 Over/Under Suggestion:** **Take the {over_under}**")
    else:
        st.warning("Insufficient data to provide betting insights for this game.")

# Visual Comparison
st.markdown("### 📊 Predicted Scores Comparison")
scores_df = pd.DataFrame({
    "Team": [home_team, away_team],
    "Predicted Score": [
        home_gbr if home_gbr is not None else 0,
        away_gbr if away_gbr is not None else 0
    ]
})

# Customize the bar chart with Seaborn
fig, ax = plt.subplots(figsize=(6, 4))
sns.set_style("whitegrid")
sns.barplot(x="Team", y="Predicted Score", data=scores_df, palette="viridis", ax=ax)
ax.set_title("Predicted Scores")
ax.set_ylabel("Score")
ax.set_ylim(0, max(scores_df["Predicted Score"].max(), 50) + 10 if scores_df["Predicted Score"].max() > 0 else 50)

# Annotate bars with values
for index, row in scores_df.iterrows():
    ax.text(index, row["Predicted Score"] + 0.5, f'{row["Predicted Score"]:.2f}', color='black', ha="center")

st.pyplot(fig)

# Additional Team Statistics (Optional)
st.markdown("### 📋 Team Statistics")
with st.expander("View Team Performance Stats"):
    stats_df = pd.DataFrame(team_stats).T
    stats_df = stats_df.rename_axis("Team").reset_index()
    st.dataframe(stats_df.style.highlight_max(axis=0), use_container_width=True)
