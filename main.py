from utils.database import init_db

# Initialize the database at startup
init_db()

import streamlit as st
import os
from importlib import import_module
from core.signal_system import signal_system
from core.state import State
from core.processing_chain import processing_chain

# Import existing static pages
from pages.monte_carlo_simulation import monte_carlo_simulation_page
from pages.correlation_analysis import correlation_analysis_page
from pages.admin_page import admin_page

# Title of the application
st.title("NFL Monte Carlo Simulation and Analysis")

# Available pages, including dynamically loaded pages
static_pages = {
    "Monte Carlo Simulation": monte_carlo_simulation_page,
    "Correlation Analysis": correlation_analysis_page,
    "Admin Panel": admin_page
}

# Dynamically load any additional pages from the 'pages' directory
def load_dynamic_pages():
    dynamic_pages = {}
    for file in os.listdir("pages"):
        if file.endswith(".py") and file not in static_pages:
            module_name = file.replace(".py", "")
            try:
                module = import_module(f"pages.{module_name}")
                if hasattr(module, module_name):
                    dynamic_pages[module_name] = getattr(module, module_name)
            except Exception as e:
                st.error(f"Failed to load page {module_name}: {e}")
    return dynamic_pages

# Combine static and dynamically loaded pages
all_pages = {**static_pages, **load_dynamic_pages()}

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a Page", list(all_pages.keys()))

# Display the selected page
if page in all_pages:
    all_pages[page]()
