import os
import streamlit as st
from pathlib import Path
from PIL import Image
import shutil
import datetime
from io import BytesIO
import base64
import json
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
    else:
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

    # Check if a post is selected for detailed view
    if st.session_state.selected_post:
        display_full_post(st.session_state.selected_post)
        return

    posts = list_posts()
    if not posts:
        st.info("No blog posts available.")
        return

    # CSS for Post Cards
    st.markdown("""
        <style>
        .post-card {
            display: flex;
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            align-items: center;
        }
        .thumbnail {
            width: 150px;
            height: 100px;
            object-fit: cover;
            border-radius: 5px;
            margin-right: 20px;
        }
        .post-details {
            flex: 1;
        }
        .post-title {
            font-size: 1.5em;
            color: #333333;
            margin-bottom: 5px;
        }
        .post-meta {
            font-size: 0.9em;
            color: #666666;
            margin-bottom: 10px;
        }
        .post-content {
            font-size: 1em;
            line-height: 1.6;
            color: #444444;
        }
        .read-more {
            display: inline-block;
            margin-top: 10px;
            font-size: 1em;
            color: #007BFF;
            text-decoration: none;
            cursor: pointer;
            transition: color 0.3s ease;
        }
        .read-more:hover {
            color: #0056b3;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display Posts in a Card Layout
    for post in posts:
        post_title = post.replace('.md', '').replace('_', ' ').title()
        post_file = POSTS_DIR / post

        # Read and process post content
        content = get_post_content(post)
        content_preview = content[:200] + "..." if len(content) > 200 else content

        # Check for associated image
        image_file = IMAGES_DIR / f"{post_file.stem}.png"
        if image_file.exists():
            img = Image.open(image_file)
            buf = BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            encoded = base64.b64encode(img_bytes).decode()
            img_html = f'<img src="data:image/png;base64,{encoded}" class="thumbnail"/>'
        else:
            img_html = '<div style="width:150px; height:100px; background-color:#cccccc; border-radius:5px; margin-right:20px;"></div>'

        pub_date = datetime.datetime.fromtimestamp(post_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        read_more_key = f"read_more_{post}"

        # Display post in a card
        with st.container():
            st.markdown(f"""
                <div class="post-card">
                    {img_html}
                    <div class="post-details">
                        <div class="post-title">{post_title}</div>
                        <div class="post-meta">Published on: {pub_date}</div>
                        <div class="post-content">{content_preview}</div>
                        <a href="#" class="read-more" id="{read_more_key}">Read More</a>
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

    post_title = post_name.replace('.md', '').replace('_', ' ').title()
    content = get_post_content(post_name)
    pub_date = datetime.datetime.fromtimestamp((POSTS_DIR / post_name).stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    st.title(post_title)
    st.markdown(f"**Published on:** {pub_date}")
    st.markdown(content)

def create_blog_post():
    st.header("📝 Create a New Blog Post")
    with st.form(key='create_post_form'):
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

        image = st.file_uploader("🖼️ Upload Thumbnail Image", type=["png", "jpg", "jpeg"])
        scheduled_date = st.date_input("📅 Schedule Date", value=datetime.date.today())
        scheduled_time = st.time_input("⏰ Schedule Time", value=datetime.time(hour=9, minute=0))
        scheduled_datetime = datetime.datetime.combine(scheduled_date, scheduled_time)
        submitted = st.form_submit_button("📤 Publish")

        if submitted:
            if (post_type == "Manual Entry" and title and content) or (post_type == "Upload PDF/HTML" and content):
                title = title or uploaded_file.name.rsplit('.', 1)[0].replace('_', ' ').title()
                filename = f"{title.replace(' ', '_').lower()}.md"
                filepath = POSTS_DIR / filename
                metadata_path = filepath.with_suffix('.json')
                image_filename = f"{filepath.stem}.png"

                if filepath.exists():
                    st.error("❌ A post with this title already exists. Please choose a different title.")
                else:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    metadata = {"scheduled_time": scheduled_datetime.isoformat()}
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f)
                    if image:
                        img = Image.open(image)
                        img.save(IMAGES_DIR / image_filename, format="PNG")
                    st.success(f"✅ Post scheduled for {scheduled_datetime}")
                    st.rerun()
            else:
                st.warning("⚠️ Please provide all required fields for the post.")

def delete_blog_posts():
    st.header("🗑️ Delete Blog Posts")
    posts = list_posts()
    if not posts:
        st.info("No blog posts available to delete.")
        return

    selected_posts = st.multiselect("Select posts to delete", posts)
    confirm_delete = st.checkbox("⚠️ I understand that deleting a post is irreversible.")

    if st.button("🗑️ Delete Selected Posts") and confirm_delete:
        for post in selected_posts:
            delete_post(post)
        st.success("✅ Posts deleted successfully.")
        st.rerun()

def main():
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
