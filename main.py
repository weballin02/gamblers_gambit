from utils.database import init_db

# Initialize the database at startup
init_db()

import streamlit as st
from pages.monte_carlo_simulation import monte_carlo_simulation_page
from pages.correlation_analysis import correlation_analysis_page
from pages.admin_page import admin_page
from core.signal_system import signal_system
from core.state import State
from core.processing_chain import processing_chain

st.title("NFL Monte Carlo Simulation and Analysis")
page = st.sidebar.selectbox("Choose a Page", ["Monte Carlo Simulation", "Correlation Analysis", "Admin Panel"])

if page == "Monte Carlo Simulation":
    monte_carlo_simulation_page()
elif page == "Correlation Analysis":
    correlation_analysis_page()
elif page == "Admin Panel":
    admin_page()
