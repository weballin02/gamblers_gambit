# blog.py

# ===========================
# 1. Import Libraries
# ===========================

import os
import streamlit as st
from pathlib import Path
from PIL import Image
import shutil
import datetime
import pytz  # Import pytz for timezone handling
from io import BytesIO
import base64
import json  # For saving scheduled metadata
import fitz  # PyMuPDF
import html2text

# ===========================
# 2. Define Directories
# ===========================

# Define directories
POSTS_DIR = Path('posts')
TRASH_DIR = Path('trash')
IMAGES_DIR = Path('images')

# Ensure directories exist
for directory in [POSTS_DIR, TRASH_DIR, IMAGES_DIR]:
    if not directory.exists():
        directory.mkdir(parents=True)

# ===========================
# 3. Streamlit App Configuration
# ===========================

st.set_page_config(
    page_title="FoxEdge - Gambler's Gambit",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================
# 5. Initialize Session State Variables
# ===========================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'selected_post' not in st.session_state:
    st.session_state.selected_post = None

# ===========================
# 6. Authentication (Basic Example)
# ===========================

def login():
    if not st.session_state.logged_in:
        st.sidebar.header("🔒 Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username == "admin" and password == "password":  # Replace with secure authentication
                st.session_state.logged_in = True
                st.sidebar.markdown('<div class="success-message">✅ Logged in successfully!</div>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown('<div class="important-alert">❌ Invalid credentials</div>', unsafe_allow_html=True)

# ===========================
# 7. Helper Functions
# ===========================

def list_posts():
    """
    List all published and scheduled posts, including those with missing metadata.
    """
    local_tz = pytz.timezone("America/Los_Angeles")
    now = datetime.datetime.now(local_tz)
    posts = []

    for post_path in POSTS_DIR.glob('*.md'):
        metadata_path = post_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as file:
                metadata = json.load(file)
                scheduled_time = datetime.datetime.fromisoformat(metadata['scheduled_time']).astimezone(local_tz)
                # Include posts that are published or currently scheduled
                if now >= scheduled_time:
                    posts.append(post_path.name)
        else:
            # Include posts without metadata as already published
            posts.append(post_path.name)

    return sorted(posts, reverse=True)

def delete_post(post_name):
    post_path = POSTS_DIR / post_name
    image_path = IMAGES_DIR / f"{post_path.stem}.png"
    metadata_path = post_path.with_suffix('.json')
    if post_path.exists():
        os.remove(post_path)
        if image_path.exists():
            os.remove(image_path)
        if metadata_path.exists():
            os.remove(metadata_path)
        return True
    return False

def get_post_content(post_name):
    post_file = POSTS_DIR / post_name
    if post_file.exists():
        with open(post_file, 'r', encoding='utf-8') as file:
            return file.read()
    return "Post content not found."

def display_full_post(post_name):
    """Display the full content of the selected post."""
    st.button("🔙 Back to Posts", on_click=lambda: st.session_state.update(selected_post=None))
    post_title = post_name.replace('.md', '').replace('_', ' ').title()
    st.header(post_title)
    content = get_post_content(post_name)
    st.markdown(content)

def process_pdf(file):
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"❌ Failed to process PDF: {e}")
        return None

def process_html(file):
    try:
        html_content = file.read().decode("utf-8")
        markdown = html2text.html2text(html_content)
        return markdown
    except Exception as e:
        st.error(f"❌ Failed to process HTML: {e}")
        return None

# ===========================
# 8. Streamlit Interface Functions
# ===========================

def view_blog_posts():
    """
    Display all published posts.
    """
    st.header("📖 Explore The Gambit")

    if st.session_state.selected_post:
        display_full_post(st.session_state.selected_post)
        return

    posts = list_posts()
    if not posts:
        st.info("No blog posts available.")
        return

    search_query = st.text_input("🔍 Search Posts", "")
    if search_query:
        posts = [post for post in posts if search_query.lower() in post.lower()]

    if not posts:
        st.warning("No posts match your search.")
        return

    for post in posts:
        post_title = post.replace('.md', '').replace('_', ' ').title()
        post_path = POSTS_DIR / post

        content_preview = get_post_content(post)[:200] + "..."
        image_path = IMAGES_DIR / f"{post_path.stem}.png"

        pub_date = datetime.datetime.fromtimestamp(post_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        read_more_key = f"read_more_{post}"

        # Check if image exists
        if image_path.exists():
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode()
            img_tag = f'<img src="data:image/png;base64,{img_base64}" width="150" height="100" />'
        else:
            img_tag = '<div style="width:150px; height:100px; background-color:#2C3E50; border-radius:5px;"></div>'

        # Card layout for each post
        st.markdown(f"""
            <div class="post-card">
                <div class="post-image">
                    {img_tag}
                </div>
                <div class="post-content">
                    <h3 class="post-title">{post_title}</h3>
                    <p class="post-date">Published on: {pub_date}</p>
                    <p class="post-preview">{content_preview}</p>
                    <button class="read-more-button" id="{read_more_key}">Read More</button>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Handle Read More button clicks
        if st.button("Read More", key=read_more_key):
            st.session_state.selected_post = post
            st.rerun()

# ===========================
# 9. Main Functionality
# ===========================

def main():
    st.sidebar.title("📂 Blog Management")
    page = st.sidebar.radio("🛠️ Choose an option", ["View Posts", "Create Post"])

    if page == "View Posts":
        view_blog_posts()

if __name__ == "__main__":
    main()
