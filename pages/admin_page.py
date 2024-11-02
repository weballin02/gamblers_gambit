
import streamlit as st
import os
from core.signal_system import signal_system

def admin_page():
    st.title("Admin Panel - Add New Pages and Utilities")

    st.subheader("Add New Page")
    new_page_name = st.text_input("Page Name (e.g., my_new_page)")
    new_page_code = st.text_area("Page Code (Enter Python code for the new page)")

    if st.button("Create Page"):
        if new_page_name and new_page_code:
            create_new_page(new_page_name, new_page_code)
            st.success(f"Page '{new_page_name}' created successfully!")

    st.subheader("Add Utility Function")
    util_name = st.text_input("Utility File Name (e.g., my_util_function)")
    util_code = st.text_area("Utility Code")

    if st.button("Add Utility"):
        if util_name and util_code:
            add_utility(util_name, util_code)
            st.success(f"Utility '{util_name}' added successfully!")

    st.subheader("Add Dependencies")
    new_dependency = st.text_input("Dependency (e.g., pandas==1.2.3)")

    if st.button("Add Dependency"):
        if new_dependency:
            add_dependency(new_dependency)
            st.success(f"Dependency '{new_dependency}' added to requirements.txt")


def create_new_page(page_name, page_code):
    page_path = f"pages/{page_name}.py"
    with open(page_path, "w") as f:
        f.write(page_code)
    signal_system.emit("on_page_added", {"page_name": page_name})

def add_utility(util_name, util_code):
    util_path = f"utils/{util_name}.py"
    with open(util_path, "w") as f:
        f.write(util_code)

def add_dependency(dependency):
    with open("requirements.txt", "a") as f:
        f.write(f"{dependency}\n")
