import os
import streamlit as st
from pathlib import Path
from PIL import Image
import shutil
import datetime
from io import BytesIO
import base64
import json  # For saving scheduled metadata
import fitz  # PyMuPDF
import html2text

# Define directories
POSTS_DIR = Path('posts')
TRASH_DIR = Path('trash')
IMAGES_DIR = Path('images')

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
    now = datetime.datetime.now()
    posts = []

    for post_path in POSTS_DIR.glob('*.md'):
        metadata_path = post_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as file:
                metadata = json.load(file)
                scheduled_time = datetime.datetime.fromisoformat(metadata['scheduled_time'])
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

# Function to process PDF files
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

# Function to process HTML files
def process_html(file):
    try:
        html_content = file.read().decode("utf-8")
        markdown = html2text.html2text(html_content)
        return markdown
    except Exception as e:
        st.error(f"❌ Failed to process HTML: {e}")
        return None

# Streamlit Interface Functions
def view_blog_posts():
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

        st.markdown(f"""
            <div class="post-card">
                <div>
                    <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" width="150" height="100" />
                </div>
                <div>
                    <h3>{post_title}</h3>
                    <p>Published on: {pub_date}</p>
                    <p>{content_preview}</p>
                    <button id="{read_more_key}">Read More</button>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Read More", key=read_more_key):
            st.session_state.selected_post = post
            st.rerun()

def display_full_post(post_name):
    st.subheader("🔙 Back to Posts")
    if st.button("← Back"):
        st.session_state.selected_post = None
        st.rerun()

    content = get_post_content(post_name)
    pub_date = datetime.datetime.fromtimestamp((POSTS_DIR / post_name).stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    st.title(post_name.replace('.md', '').replace('_', ' ').title())
    st.markdown(f"**Published on:** {pub_date}")
    st.markdown(content)

def create_blog_post():
    st.header("📝 Create a New Blog Post")
    with st.form(key='create_post_form'):
        post_type = st.radio("Choose Post Creation Method", ["Manual Entry", "Upload PDF/HTML"], horizontal=True)
        title = st.text_input("🖊️ Post Title", placeholder="Enter the title of your post")
        content = st.text_area("📝 Content", height=300) if post_type == "Manual Entry" else None

        if post_type == "Upload PDF/HTML":
            uploaded_file = st.file_uploader("📂 Upload PDF or HTML File", type=["pdf", "html"])
            if uploaded_file:
                content = process_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else process_html(uploaded_file)

        image = st.file_uploader("🖼️ Upload Thumbnail Image", type=["png", "jpg", "jpeg"])
        scheduled_date = st.date_input("📅 Schedule Date", value=datetime.date.today())
        scheduled_time = st.time_input("⏰ Schedule Time", value=datetime.time(9, 0))
        scheduled_datetime = datetime.datetime.combine(scheduled_date, scheduled_time)
        submitted = st.form_submit_button("📤 Publish")

        if submitted and title and content:
            filename = f"{title.replace(' ', '_').lower()}.md"
            filepath = POSTS_DIR / filename
            metadata_path = filepath.with_suffix('.json')

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({"scheduled_time": scheduled_datetime.isoformat()}, f)

            if image:
                img = Image.open(image)
                img.save(IMAGES_DIR / f"{filepath.stem}.png", format="PNG")

            st.success(f"✅ Post scheduled for {scheduled_datetime}")
            st.rerun()

def delete_blog_posts():
    st.header("🗑️ Delete Blog Posts")
    posts = list_posts()
    selected_posts = st.multiselect("Select posts to delete", posts)
    confirm_delete = st.checkbox("⚠️ Confirm Deletion")

    if st.button("Delete Selected") and confirm_delete:
        for post in selected_posts:
            delete_post(post)
        st.success("✅ Deleted successfully")
        st.rerun()

def main():
    st.sidebar.title("📂 Blog Management")
    page = st.sidebar.radio("🛠️ Choose an option", ["View Posts", "Create Post", "Delete Post"])
    if page == "View Posts":
        view_blog_posts()
    elif page in ["Create Post", "Delete Post"]:
        login()
        if st.session_state.logged_in:
            create_blog_post() if page == "Create Post" else delete_blog_posts()

if __name__ == "__main__":
    main()
