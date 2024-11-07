# admin_page.py

import streamlit as st
import os
import re
from core.signal_system import signal_system

# Helper Functions

def create_new_page(page_name, page_code):
    """Creates a new page in the pages directory with the given code."""
    page_path = f"pages/{page_name}.py"
    os.makedirs("pages", exist_ok=True)  # Ensure pages directory exists

    # Check if the page already exists
    if os.path.exists(page_path):
        return False, f"Page '{page_name}' already exists."

    # Write the page code to a new file
    try:
        with open(page_path, "w") as f:
            f.write(page_code)
        signal_system.emit("on_page_added", {"page_name": page_name})  # Signal that a new page was added
        return True, f"Page '{page_name}' created successfully!"
    except Exception as e:
        return False, f"Error creating page '{page_name}': {e}"

def add_utility(util_name, util_code):
    """Adds a new utility file with the provided code in the utils directory."""
    util_path = f"utils/{util_name}.py"
    os.makedirs("utils", exist_ok=True)  # Ensure utils directory exists

    # Check if the utility already exists
    if os.path.exists(util_path):
        return False, f"Utility '{util_name}' already exists."

    # Write the utility code to a new file
    try:
        with open(util_path, "w") as f:
            f.write(util_code)
        return True, f"Utility '{util_name}' added successfully!"
    except Exception as e:
        return False, f"Error adding utility '{util_name}': {e}"

def add_dependency(dependency):
    """Adds a new dependency to the requirements.txt file if it doesn't already exist."""
    requirements_path = "requirements.txt"
    os.makedirs(os.path.dirname(requirements_path), exist_ok=True)  # Ensure directory exists

    # Check if dependency already exists
    with open(requirements_path, "r") as f:
        existing_dependencies = f.read().splitlines()

    if dependency in existing_dependencies:
        return False, f"Dependency '{dependency}' is already in requirements.txt."

    # Append the new dependency
    try:
        with open(requirements_path, "a") as f:
            f.write(f"{dependency}\n")
        return True, f"Dependency '{dependency}' added to requirements.txt."
    except Exception as e:
        return False, f"Error adding dependency '{dependency}': {e}"

def extract_functions(page_code):
    """
    Extracts function definitions from the provided page code.
    Returns a dictionary with function names as keys and code as values.
    """
    function_pattern = re.compile(r"def (\w+)\(.*\):(.+?)(?=\ndef|\Z)", re.DOTALL)
    matches = function_pattern.findall(page_code)
    return {name: f"def {name}(...):{body.strip()}" for name, body in matches}

def detect_utility_candidates(functions):
    """
    Detects utility candidates based on keywords and patterns.
    Returns a list of function names that are likely to be reusable.
    """
    keywords = ["fetch", "process", "calculate", "load", "save", "util", "helper"]
    utility_candidates = {}

    for func_name, func_code in functions.items():
        # Flag as utility candidate if function name contains common keywords
        if any(keyword in func_name.lower() for keyword in keywords):
            utility_candidates[func_name] = func_code

    return utility_candidates

def move_to_utils(selected_functions, extracted_functions, page_name, page_code):
    """
    Moves selected functions to `utils.py`, updates page code with imports,
    and returns the updated page code.
    """
    utils_path = "utils/utils.py"
    os.makedirs(os.path.dirname(utils_path), exist_ok=True)

    # Append selected functions to `utils.py`
    with open(utils_path, "a") as utils_file:
        for func_name in selected_functions:
            utils_file.write("\n\n" + extracted_functions[func_name])

    # Generate import statements for each selected function
    imports = "\n".join([f"from utils import {func_name}" for func_name in selected_functions])

    # Remove the original function definitions from the page code
    for func_name in selected_functions:
        page_code = re.sub(rf"def {func_name}\(.*\):(.+?)(?=\ndef|\Z)", "", page_code, flags=re.DOTALL).strip()

    # Prepend imports to the updated page code
    updated_code = imports + "\n\n" + page_code

    # Rewrite the page with updated code
    page_path = f"pages/{page_name}.py"
    with open(page_path, "w") as page_file:
        page_file.write(updated_code)

    return updated_code  # Return the updated code to display to the user


# Main Admin UI

# Title and introductory text
st.title("Admin Panel - Add New Pages and Utilities")
st.write("Use this panel to add new pages, utilities, and dependencies to your app.")

# Section to add a new page
st.subheader("Add New Page")
new_page_name = st.text_input("Page Name (e.g., my_new_page)")
new_page_code = st.text_area("Page Code (Enter Python code for the new page)")

if st.button("Create Page"):
    if new_page_name and new_page_code:
        success, message = create_new_page(new_page_name, new_page_code)
        if success:
            st.success(message)
            st.info("Would you like to generate utility functions from this page code?")

            # Automatically generate utilities
            if st.button("Generate Utilities from Page Code"):
                extracted_functions = extract_functions(new_page_code)
                
                if extracted_functions:
                    utility_candidates = detect_utility_candidates(extracted_functions)
                    if utility_candidates:
                        st.write("Detected potential utility functions based on patterns:")
                        selected_functions = st.multiselect(
                            "Select functions to move to `utils.py`",
                            options=list(utility_candidates.keys()),
                            default=list(utility_candidates.keys())
                        )

                        if st.button("Move Selected to `utils.py`"):
                            updated_code = move_to_utils(selected_functions, extracted_functions, new_page_name, new_page_code)
                            st.success("Selected functions moved to `utils.py` and page code updated!")
                            st.code(updated_code, language="python")  # Display the updated page code
                    else:
                        st.info("No reusable functions detected based on common patterns.")
                else:
                    st.info("No functions detected in the provided code.")
    else:
        st.warning("Please enter both a page name and code to create a new page.")

# Section to add a new utility function
st.subheader("Add Utility Function")
util_name = st.text_input("Utility File Name (e.g., my_util_function)")
util_code = st.text_area("Utility Code")

if st.button("Add Utility"):
    if util_name and util_code:
        success, message = add_utility(util_name, util_code)
        if success:
            st.success(message)
        else:
            st.error(message)
    else:
        st.warning("Please enter both a utility file name and code to add a utility.")

# Section to add a new dependency
st.subheader("Add Dependency")
new_dependency = st.text_input("Dependency (e.g., pandas==1.2.3)")

if st.button("Add Dependency"):
    if new_dependency:
        success, message = add_dependency(new_dependency)
        if success:
            st.success(message)
        else:
            st.warning(message)
    else:
        st.warning("Please enter a dependency to add.")
