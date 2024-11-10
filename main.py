import streamlit as st
from utils.database import init_db

# Initialize the database at startup
init_db()

# Title and introduction for the main landing page
st.title("Welcome to Gambler's Gambit")
st.write("Discover a powerful app for predicting NBA and NFL games, using real-time data and advanced simulations to give you an edge. This overview highlights all the features so you know exactly where to go for game predictions, score analysis, and betting insights.")
