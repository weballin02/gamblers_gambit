import os
import json
import streamlit as st
from pathlib import Path
from PIL import Image
import shutil
import datetime
from datetime import datetime as dt
from io import BytesIO
import base64
import fitz  # PyMuPDF
import html2text

# Define directories
POSTS_DIR = Path('posts')
TRASH_DIR = Path('trash')
IMAGES_DIR = Path('images')  # Directory for images
METADATA_DIR = Path('metadata')  # Directory for metadata

# Ensure directories exist
for directory in [POSTS_DIR, TRASH_DIR, IMAGES_DIR, METADATA_DIR]:
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
def save_metadata(post_name, scheduled_time):
    """Save metadata for a post, including the scheduled publish time."""
    metadata_file = METADATA_DIR / f"{post_name}.json"
    metadata = {"scheduled_time": scheduled_time.isoformat()}
    with open(metadata_file, 'w', encoding='utf-8') as file:
        json.dump(metadata, file)

def load_metadata(post_name):
    """Load metadata for a post, including the scheduled publish time."""
    metadata_file = METADATA_DIR / f"{post_name}.json"
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    return None

def list_posts():
    """List posts visible based on their scheduled publish time."""
    posts = sorted([f.name for f in POSTS_DIR.glob('*.md')], reverse=True)
    visible_posts = []
    for post in posts:
        metadata = load_metadata(post.replace('.md', ''))
        if metadata:
            scheduled_time = dt.fromisoformat(metadata['scheduled_time'])
            if dt.now() >= scheduled_time:
                visible_posts.append(post)
        else:
            # If no metadata, consider the post immediately visible
            visible_posts.append(post)
    return visible_posts

def delete_post(post_name):
    post_path = POSTS_DIR / post_name
    image_path = IMAGES_DIR / f"{post_path.stem}.png"
    metadata_path = METADATA_DIR / f"{post_path.stem}.json"
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
    metadata_path = METADATA_DIR / f"{post_path.stem}.json"
    trash_metadata_path = TRASH_DIR / f"{post_path.stem}.json"

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
        return html2text.html2text(html_content)
    except Exception as e:
        st.error(f"❌ Failed to process HTML: {e}")
        return None

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
    filtered_posts = [post for post in posts if search_query.lower() in post.lower()] if search_query else posts

    if not filtered_posts:
        st.warning("No posts match your search.")
        return

    for post in filtered_posts:
        post_title = post.replace('.md', '').replace('_', ' ').title()
        post_file = POSTS_DIR / post
        content_preview = get_post_content(post)[:200] + "..."
        pub_date = datetime.datetime.fromtimestamp(post_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        read_more_key = f"read_more_{post}"

        with st.container():
            st.markdown(f"""
                **{post_title}**  
                *Published on: {pub_date}*  
                {content_preview}  
            """)
            if st.button("Read More", key=read_more_key):
                st.session_state.selected_post = post
                st.rerun()

def display_full_post(post_name):
    st.subheader("🔙 Back to Posts")
    if st.button("← Back"):
        st.session_state.selected_post = None
        st.rerun()

    post_title = post_name.replace('.md', '').replace('_', ' ').title()
    post_file = POSTS_DIR / post_name
    content = get_post_content(post_name)

    st.title(post_title)
    pub_date = datetime.datetime.fromtimestamp(post_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    st.markdown(f"**Published on:** {pub_date}")

    image_file = IMAGES_DIR / f"{post_file.stem}.png"
    if image_file.exists():
        st.image(str(image_file), use_column_width=True)

    st.markdown(content)

def create_blog_post():
    st.header("📝 Create a New Blog Post")
    with st.form(key='create_post_form'):
        title = st.text_input("🖊️ Post Title", placeholder="Enter the title of your post")
        content = st.text_area("📝 Content", height=300, placeholder="Write your post content here...")
        image = st.file_uploader("🖼️ Upload Thumbnail Image", type=["png", "jpg", "jpeg"])
        
        # Add date and time inputs for scheduling
        post_date = st.date_input("📅 Schedule Date", value=datetime.date.today())
        post_time = st.time_input("⏰ Schedule Time", value=datetime.time(hour=9, minute=0))
        
        # Combine date and time into a single datetime object
        scheduled_time = datetime.datetime.combine(post_date, post_time)

        submitted = st.form_submit_button("📤 Publish")

        if submitted:
            if title and content:
                filename = f"{title.replace(' ', '_').lower()}.md"
                filepath = POSTS_DIR / filename
                image_filename = f"{filepath.stem}.png"
                image_path = IMAGES_DIR / image_filename

                if filepath.exists():
                    st.error("❌ A post with this title already exists. Please choose a different title.")
                else:
                    # Save the markdown file
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(content)

                    # Save the uploaded image if provided
                    if image:
                        img = Image.open(image)
                        img.save(image_path, format="PNG")

                    # Save metadata with the scheduled time
                    save_metadata(filepath.stem, scheduled_time)
                    st.success(f"✅ Post scheduled for: **{scheduled_time}**")
                    st.rerun()
            else:
                st.warning("⚠️ Please provide both a title and content for the post.")

def delete_blog_posts():
    st.header("🗑️ Delete Blog Posts")

    posts = list_posts()
    if not posts:
        st.info("No blog posts available to delete.")
        return

    confirm_delete = st.checkbox("⚠️ Confirm deletion.")
    selected_posts = st.multiselect("Select posts to delete", posts)

    if st.button("🗑️ Delete Selected Posts"):
        if confirm_delete:
            for post in selected_posts:
                if move_to_trash(post):
                    st.success(f"✅ Moved to trash: {post.replace('.md', '').title()}")
            st.rerun()
        else:
            st.warning("⚠️ Please confirm deletion.")

def display_header():
    st.title("📝 Gambler's Gambit")

def main():
    display_header()

    st.sidebar.title("📂 Blog Management")
    page = st.sidebar.radio("🛠️ Choose an option", ["View Posts", "Create Post", "Delete Post"])

    if page == "View Posts":
        view_blog_posts()
    elif page in ["Create Post", "Delete Post"]:
        login()
        if st.session_state.logged_in:
            if page == "Create Post":
                create_blog_post()
            elif page == "Delete Post":
                delete_blog_posts()
        else:
            st.warning("🔒 Please log in.")

if __name__ == "__main__":
    main()
