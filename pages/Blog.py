import os
import streamlit as st
from pathlib import Path
from PIL import Image
import shutil
import datetime
import json
from io import BytesIO
import base64

# New imports for handling PDF and HTML
import fitz  # PyMuPDF
import html2text

# Define directories
POSTS_DIR = Path('posts')
TRASH_DIR = Path('trash')
IMAGES_DIR = Path('images')  # Directory for images

# Ensure directories exist
for directory in [POSTS_DIR, TRASH_DIR, IMAGES_DIR]:
    if not directory.exists():
        directory.mkdir(parents=True)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Gambler's Gambit",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'selected_post' not in st.session_state:
    st.session_state.selected_post = None

# Authentication (Basic Example)
def login():
    if not st.session_state.logged_in:
        st.sidebar.header("🔒 Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username == "admin" and password == "password":  # Replace with secure authentication
                st.session_state.logged_in = True
                st.sidebar.success("✅ Logged in successfully!")
            else:
                st.sidebar.error("❌ Invalid credentials")

# Helper Functions
def list_posts():
    posts = sorted([f.name for f in POSTS_DIR.glob('*.md')], reverse=True)
    return posts

def list_scheduled_posts():
    metadata_files = sorted(POSTS_DIR.glob('*.json'), key=lambda f: f.stat().st_mtime, reverse=True)
    scheduled_posts = []
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as file:
                metadata = json.load(file)
            if "scheduled_time" in metadata:
                scheduled_time = datetime.datetime.fromisoformat(metadata["scheduled_time"])
                if scheduled_time > datetime.datetime.now():
                    scheduled_posts.append({
                        "title": metadata_file.stem.replace('_', ' ').title(),
                        "scheduled_time": scheduled_time,
                        "metadata_file": metadata_file
                    })
        except Exception as e:
            st.error(f"❌ Error reading metadata: {e}")
    return scheduled_posts

def get_post_content(post_name):
    post_file = POSTS_DIR / post_name
    if post_file.exists():
        with open(post_file, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    return "Post content not found."

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

# Display scheduled posts
def view_scheduled_posts():
    st.header("📅 Scheduled Articles")
    scheduled_posts = list_scheduled_posts()
    if not scheduled_posts:
        st.info("No articles are scheduled for publication.")
        return

    for post in scheduled_posts:
        st.markdown(f"""
            ### {post['title']}
            **Scheduled Time:** {post['scheduled_time'].strftime('%Y-%m-%d %H:%M')}
        """)

# View existing blog posts
def view_blog_posts():
    st.header("📖 Explore The Gambit")
    posts = list_posts()
    if not posts:
        st.info("No blog posts available.")
        return

    # Search Functionality
    search_query = st.text_input("🔍 Search Posts", "")
    filtered_posts = [post for post in posts if search_query.lower() in post.lower()] if search_query else posts

    if not filtered_posts:
        st.warning("No posts match your search.")
        return

    for post in filtered_posts:
        title = post.replace('.md', '').replace('_', ' ').title()
        content = get_post_content(post)
        content_preview = content[:200] + "..." if len(content) > 200 else content

        st.markdown(f"### {title}")
        st.text(content_preview)
        st.markdown("---")

# Create blog posts
def create_blog_post():
    st.header("📝 Create a New Blog Post")
    post_type = st.radio("Choose Post Creation Method", ["Manual Entry", "Upload PDF/HTML"], horizontal=True)

    if post_type == "Manual Entry":
        title = st.text_input("🖊️ Post Title", placeholder="Enter the title of your post")
        content = st.text_area("📝 Content", height=300, placeholder="Write your post content here...")
    elif post_type == "Upload PDF/HTML":
        uploaded_file = st.file_uploader("📂 Upload PDF or HTML File", type=["pdf", "html"])
        title = st.text_input("🖊️ Post Title (Optional)", placeholder="Enter the title of your post (optional)")
        content = None
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                content = process_pdf(uploaded_file)
            elif uploaded_file.type in ["text/html", "application/xhtml+xml"]:
                content = process_html(uploaded_file)
            else:
                st.error("❌ Unsupported file type.")

    image = st.file_uploader("🖼️ Upload Thumbnail Image", type=["png", "jpg", "jpeg"])
    scheduled_date = st.date_input("📅 Schedule Date", value=datetime.date.today())
    scheduled_time = st.time_input("⏰ Schedule Time", value=datetime.time(9, 0))
    scheduled_datetime = datetime.datetime.combine(scheduled_date, scheduled_time)

    if st.button("📤 Publish"):
        if post_type == "Manual Entry" and (not title or not content):
            st.warning("⚠️ Please provide both a title and content for the post.")
            return
        elif post_type == "Upload PDF/HTML" and not uploaded_file:
            st.warning("⚠️ Please upload a PDF or HTML file to create a post.")
            return

        if not title:
            title = uploaded_file.name.rsplit('.', 1)[0].replace('_', ' ').title()

        filename = f"{title.replace(' ', '_').lower()}.md"
        filepath = POSTS_DIR / filename
        metadata_path = filepath.with_suffix('.json')

        if filepath.exists():
            st.error("❌ A post with this title already exists. Please choose a different title.")
            return

        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)

        with open(metadata_path, 'w', encoding='utf-8') as file:
            json.dump({"scheduled_time": scheduled_datetime.isoformat()}, file)

        if image:
            try:
                img = Image.open(image)
                img.save(IMAGES_DIR / f"{filepath.stem}.png", format="PNG")
                st.success(f"✅ Published post with image: **{title}** scheduled for {scheduled_datetime}")
            except Exception as e:
                st.error(f"❌ Failed to save image: {e}")
        else:
            st.success(f"✅ Published post: **{title}** (No image uploaded) scheduled for {scheduled_datetime}")

        st.rerun()

# Main Function
def main():
    st.sidebar.title("📂 Blog Management")
    page = st.sidebar.radio("🛠️ Choose an option", ["View Posts", "Create Post", "View Scheduled Posts"])

    if page == "View Posts":
        view_blog_posts()
    elif page == "Create Post":
        login()
        if st.session_state.logged_in:
            create_blog_post()
        else:
            st.warning("🔒 Please log in to access this feature.")
    elif page == "View Scheduled Posts":
        login()
        if st.session_state.logged_in:
            view_scheduled_posts()
        else:
            st.warning("🔒 Please log in to access this feature.")

if __name__ == "__main__":
    main()
