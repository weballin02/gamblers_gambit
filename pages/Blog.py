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
import pytz
import json
from io import BytesIO
import base64
import fitz
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
    page_title="FoxEdge - Blog Management",
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
                st.sidebar.success("Logged in successfully!")
            else:
                st.sidebar.error("Invalid credentials")

# ===========================
# 7. Helper Functions
# ===========================

def list_posts():
    local_tz = pytz.timezone("America/Los_Angeles")
    now = datetime.datetime.now(local_tz)
    posts = []

    for post_path in POSTS_DIR.glob('*.md'):
        metadata_path = post_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as file:
                metadata = json.load(file)
                scheduled_time = datetime.datetime.fromisoformat(metadata['scheduled_time']).astimezone(local_tz)
                if now >= scheduled_time:
                    posts.append(post_path.name)
        else:
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

def move_to_trash(post_name):
    post_path = POSTS_DIR / post_name
    trash_post_path = TRASH_DIR / post_name
    image_path = IMAGES_DIR / f"{post_path.stem}.png"
    trash_image_path = TRASH_DIR / f"{post_path.stem}.png"

    metadata_path = post_path.with_suffix('.json')
    trash_metadata_path = TRASH_DIR / metadata_path.name

    if post_path.exists():
        post_path.rename(trash_post_path)
        if image_path.exists():
            image_path.rename(trash_image_path)
        if metadata_path.exists():
            metadata_path.rename(trash_metadata_path)
        return True
    return False

def get_post_content(post_name):
    post_file = POSTS_DIR / post_name
    if post_file.exists():
        with open(post_file, 'r', encoding='utf-8') as file:
            return file.read()
    return "Post content not found."

def display_full_post(post_name):
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
    st.header("📖 Explore Posts")

    if st.session_state.selected_post:
        display_full_post(st.session_state.selected_post)
        return

    posts = list_posts()
    if not posts:
        st.info("No blog posts available.")
        return

    for post in posts:
        post_title = post.replace('.md', '').replace('_', ' ').title()
        post_path = POSTS_DIR / post
        content_preview = get_post_content(post)[:200] + "..."
        pub_date = datetime.datetime.fromtimestamp(post_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')

        if st.button(f"Read More: {post_title}"):
            st.session_state.selected_post = post
            st.rerun()

def create_blog_post():
    st.header("📝 Create a New Post")
    title = st.text_input("Post Title")
    content = st.text_area("Content", height=300)
    image = st.file_uploader("Upload Thumbnail Image", type=["png", "jpg", "jpeg"])
    scheduled_date = st.date_input("Schedule Date", value=datetime.date.today())
    scheduled_time = st.time_input("Schedule Time", value=datetime.time(9, 0))
    local_tz = pytz.timezone("America/Los_Angeles")
    scheduled_datetime = local_tz.localize(datetime.datetime.combine(scheduled_date, scheduled_time))

    if st.button("Publish"):
        if not title or not content:
            st.warning("Title and content are required.")
            return

        filename = f"{title.replace(' ', '_').lower()}.md"
        filepath = POSTS_DIR / filename
        metadata_path = filepath.with_suffix('.json')

        if filepath.exists():
            st.error("A post with this title already exists.")
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(content)

            with open(metadata_path, 'w', encoding='utf-8') as file:
                json.dump({"scheduled_time": scheduled_datetime.isoformat()}, file)

            if image:
                img = Image.open(image)
                img.save(IMAGES_DIR / f"{filepath.stem}.png", format="PNG")

            st.success(f"Published post: {title}")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to publish post: {e}")

def edit_blog_post():
    posts = list_posts()
    selected_post = st.selectbox("Select a post to edit", posts)

    if not selected_post:
        st.warning("Select a post to edit.")
        return

    post_path = POSTS_DIR / selected_post
    metadata_path = post_path.with_suffix('.json')

    content = get_post_content(selected_post)
    with open(metadata_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    scheduled_time = datetime.datetime.fromisoformat(metadata['scheduled_time'])
    scheduled_date = scheduled_time.date()
    scheduled_time_input = scheduled_time.time()

    title = st.text_input("Post Title", value=selected_post.replace('_', ' ').replace('.md', ''))
    updated_content = st.text_area("Content", value=content, height=300)
    new_date = st.date_input("Schedule Date", value=scheduled_date)
    new_time = st.time_input("Schedule Time", value=scheduled_time_input)

    local_tz = pytz.timezone("America/Los_Angeles")
    updated_scheduled_time = local_tz.localize(datetime.datetime.combine(new_date, new_time))

    if st.button("Update Post"):
        with open(post_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)

        with open(metadata_path, 'w', encoding='utf-8') as file:
            json.dump({"scheduled_time": updated_scheduled_time.isoformat()}, file)

        st.success("Post updated successfully!")
        st.rerun()

def delete_blog_posts():
    posts = list_posts()
    selected_posts = st.multiselect("Select posts to delete", posts)

    if st.button("Delete Selected"):
        for post in selected_posts:
            delete_post(post)
        st.success("Selected posts deleted successfully.")
        st.rerun()

def view_scheduled_posts():
    st.header("📅 Scheduled Posts")
    local_tz = pytz.timezone("America/Los_Angeles")
    now = datetime.datetime.now(local_tz)

    scheduled_posts = []
    for metadata_path in POSTS_DIR.glob('*.json'):
        with open(metadata_path, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
            scheduled_time = datetime.datetime.fromisoformat(metadata['scheduled_time']).astimezone(local_tz)
            if scheduled_time > now:
                scheduled_posts.append((metadata_path.stem, scheduled_time))

    if not scheduled_posts:
        st.info("No scheduled posts available.")
        return

    for post_name, scheduled_time in scheduled_posts:
        st.markdown(f"**{post_name.replace('_', ' ').title()}** - Scheduled for: {scheduled_time.strftime('%Y-%m-%d %H:%M')}")

# ===========================
# 9. Main Functionality
# ===========================

def main():
    st.sidebar.title("Blog Management")
    page = st.sidebar.radio("Choose an option", ["View Posts", "Create Post", "Edit Post", "Delete Posts", "Scheduled Posts"])

    if page == "View Posts":
        view_blog_posts()
    elif page == "Create Post":
        create_blog_post()
    elif page == "Edit Post":
        edit_blog_post()
    elif page == "Delete Posts":
        delete_blog_posts()
    elif page == "Scheduled Posts":
        view_scheduled_posts()

if __name__ == "__main__":
    main()
