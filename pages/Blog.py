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
# 4. Custom CSS Styling with FoxEdge Colors and Enhancements
# ===========================

st.markdown('''
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&family=Open+Sans:wght@400;600&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');

        /* Root Variables */
        :root {
            --background-gradient-start: #2C3E50; /* Charcoal Dark Gray */
            --background-gradient-end: #1E90FF;   /* Electric Blue */
            --primary-text-color: #FFFFFF;         /* Crisp White */
            --heading-text-color: #F5F5F5;         /* Light Gray */
            --accent-color-teal: #32CD32;          /* Lime Green */
            --accent-color-purple: #FF8C00;        /* Deep Orange */
            --highlight-color: #FFFF33;            /* Neon Yellow */
            --font-heading: 'Raleway', sans-serif;
            --font-body: 'Open Sans', sans-serif;
            --font-montserrat: 'Montserrat', sans-serif;
        }

        /* Global Styles */
        body, html {
            background: linear-gradient(135deg, var(--background-gradient-start), var(--background-gradient-end));
            color: var(--primary-text-color);
            font-family: var(--font-body);
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }

        h1, h2, h3, h4 {
            font-family: var(--font-heading);
            color: var(--heading-text-color);
        }

        /* Hero Section */
        .hero {
            position: relative;
            text-align: center;
            padding: 4em 1em;
            overflow: hidden;
        }

        .hero::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at center, rgba(255, 255, 255, 0.1), transparent);
            animation: rotate 30s linear infinite;
        }

        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .hero h1 {
            font-size: 3.5em;
            margin-bottom: 0.2em;
            background: linear-gradient(120deg, var(--accent-color-teal), var(--accent-color-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: var(--font-montserrat);
        }

        .hero p {
            font-size: 1.5em;
            margin-bottom: 1em;
            color: #CCCCCC; /* Light Gray */
        }

        /* Buttons */
        .button {
            background: linear-gradient(45deg, var(--accent-color-teal), var(--accent-color-purple));
            border: none;
            padding: 0.8em 2em;
            color: #FFFFFF; /* Crisp White */
            font-size: 1.1em;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.3s ease, background 0.3s ease;
            text-decoration: none;
            display: inline-block;
            margin-top: 1em;
            font-family: var(--font-montserrat);
        }

        .button:hover {
            transform: translateY(-5px);
            background: linear-gradient(45deg, var(--accent-color-purple), var(--accent-color-teal));
        }

        /* Select Box Styling */
        .css-1aumxhk {
            background-color: rgba(44, 62, 80, 0.8); /* Semi-transparent Charcoal Dark Gray */
            color: #FFFFFF; /* Crisp White */
            border: 1px solid #1E90FF; /* Electric Blue */
            border-radius: 5px;
        }

        /* Select Box Option Styling */
        .css-1y4p8pa {
            color: #FFFFFF; /* Crisp White */
            background-color: rgba(44, 62, 80, 0.8); /* Semi-transparent Charcoal Dark Gray */
        }

        /* Post Card Styling */
        .post-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            background-color: rgba(44, 62, 80, 0.8); /* Semi-transparent Charcoal Dark Gray */
        }

        .post-image {
            flex: 0 0 150px;
        }

        .post-image img {
            width: 150px;
            height: 100px;
            border-radius: 5px;
        }

        .post-content {
            flex: 1;
            padding-left: 10px;
        }

        .post-title {
            margin: 0;
            font-family: var(--font-montserrat);
            font-size: 1.5em;
            background: linear-gradient(120deg, var(--accent-color-teal), var(--accent-color-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .post-date {
            color: #CCCCCC; /* Light Gray */
            margin: 5px 0;
            font-size: 0.9em;
        }

        .post-preview {
            color: #FFFFFF; /* Crisp White */
        }

        .read-more-button {
            background: #1E90FF; /* Electric Blue */
            color: #FFFFFF; /* Crisp White */
            border: none;
            border-radius: 5px;
            padding: 5px 10px;
            cursor: pointer;
            transition: background 0.3s ease;
            font-family: var(--font-montserrat);
        }

        .read-more-button:hover {
            background: #32CD32; /* Lime Green */
        }

        /* Summary and Prediction Results Sections */
        .summary-section, .results-section {
            padding: 2em 1em;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            margin-bottom: 2em;
        }

        .summary-section h3, .results-section h3 {
            font-size: 2em;
            margin-bottom: 0.5em;
            color: var(--accent-color-purple); /* Deep Orange */
        }

        /* Success Message Styling */
        .success-message {
            background-color: #32CD32; /* Lime Green */
            color: #FFFFFF; /* Crisp White */
            padding: 0.5em;
            border-radius: 5px;
            font-weight: bold;
            margin-bottom: 1em;
        }

        /* Important Alert Styling */
        .important-alert {
            background-color: #FFFF33; /* Neon Yellow */
            color: #2C3E50; /* Charcoal Dark Gray for text */
            padding: 0.5em;
            border-radius: 5px;
            font-weight: bold;
            margin-bottom: 1em;
        }

        /* Electric Blue for Links */
        a {
            color: #1E90FF; /* Electric Blue */
            text-decoration: none;
            font-weight: 600;
        }

        a:hover {
            text-decoration: underline;
        }

        /* Footer Styling */
        .footer {
            text-align: center;
            padding: 2em 1em;
            color: #999999;
            font-size: 0.9em;
        }

        .footer a {
            color: #32CD32; /* Lime Green */
            text-decoration: none;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .post-card {
                flex-direction: column;
                align-items: center;
            }

            .post-content {
                padding-left: 0;
                text-align: center;
            }

            .metric-container {
                flex-direction: column;
                align-items: center;
            }

            .metric {
                width: 90%;
            }
        }

    </style>
''', unsafe_allow_html=True)

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
    # Get current time in user's local timezone (Pacific Time)
    local_tz = pytz.timezone("America/Los_Angeles")  # Change this to the user's local timezone
    now = datetime.datetime.now(local_tz)
    posts = []

    for post_path in POSTS_DIR.glob('*.md'):
        metadata_path = post_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as file:
                metadata = json.load(file)
                scheduled_time = datetime.datetime.fromisoformat(metadata['scheduled_time']).astimezone(local_tz)
                # Only include posts that are already published
                if now >= scheduled_time:
                    posts.append(post_path.name)
        else:
            posts.append(post_path.name)  # Include already published posts

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
    """Display the full content of the selected post."""
    st.button("🔙 Back to Posts", on_click=lambda: st.session_state.update(selected_post=None))
    post_title = post_name.replace('.md', '').replace('_', ' ').title()
    st.header(post_title)
    content = get_post_content(post_name)
    st.markdown(content)

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

# ===========================
# 8. Streamlit Interface Functions
# ===========================

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
            st.experimental_rerun()

def create_blog_post():
    st.header("📝 Create a New Blog Post")
    post_type = st.selectbox("Select Post Type", ["Manual Entry", "Upload PDF/HTML"])

    if post_type == "Manual Entry":
        title = st.text_input("🖊️ Post Title")
        content = st.text_area("📝 Content", height=300)
    elif post_type == "Upload PDF/HTML":
        uploaded_file = st.file_uploader("📂 Upload PDF or HTML File", type=["pdf", "html"])
        title = st.text_input("🖊️ Post Title (Optional)", placeholder="Enter the title of your post (optional)")
        content = None  # Will be populated after processing the file

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

    # Convert to user's local timezone (Pacific Time)
    local_tz = pytz.timezone("America/Los_Angeles")  # Change this to the user's local timezone
    scheduled_datetime = local_tz.localize(scheduled_datetime)

    if st.button("📤 Publish"):
        if post_type == "Manual Entry" and (not title or not content):
            st.warning("⚠️ Please provide both a title and content for the post.")
            return
        elif post_type == "Upload PDF/HTML" and not uploaded_file:
            st.warning("⚠️ Please upload a PDF or HTML file to create a post.")
            return

        # Set title from file name if not provided
        if not title and post_type == "Upload PDF/HTML":
            title = uploaded_file.name.rsplit('.', 1)[0].replace('_', ' ').title()

        filename = f"{title.replace(' ', '_').lower()}.md"
        filepath = POSTS_DIR / filename
        metadata_path = filepath.with_suffix('.json')

        if filepath.exists():
            st.error("❌ A post with this title already exists. Please choose a different title.")
            return

        # Save the markdown file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)

        # Save scheduling metadata
        with open(metadata_path, 'w', encoding='utf-8') as file:
            json.dump({"scheduled_time": scheduled_datetime.isoformat()}, file)

        # Save the uploaded image if provided
        if image:
            try:
                img = Image.open(image)
                img.save(IMAGES_DIR / f"{filepath.stem}.png", format="PNG")
                st.markdown(f'<div class="success-message">✅ Published post with image: **{title}** scheduled for {scheduled_datetime}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Failed to save image: {e}")
        else:
            st.markdown(f'<div class="success-message">✅ Published post: **{title}** (No image uploaded) scheduled for {scheduled_datetime}</div>', unsafe_allow_html=True)

        st.experimental_rerun()

def edit_scheduled_post():
    if not st.session_state.logged_in:
        st.warning("⚠️ You must be logged in to edit scheduled posts.")
        return

    st.header("✏️ Edit Scheduled Post")
    # Get all posts, including scheduled ones
    all_posts = list_posts()  # This will only include already published posts
    scheduled_posts = []

    # Collect scheduled posts
    for post_path in POSTS_DIR.glob('*.json'):
        with open(post_path, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
            scheduled_time = datetime.datetime.fromisoformat(metadata['scheduled_time']).astimezone(pytz.timezone("America/Los_Angeles"))
            if scheduled_time > datetime.datetime.now(pytz.timezone("America/Los_Angeles")):
                scheduled_posts.append(post_path.stem)  # Add the post name without extension

    # Combine both lists for selection
    all_posts.extend(scheduled_posts)
    selected_post = st.selectbox("Select a post to edit", all_posts)  # Show all posts including scheduled ones

    if selected_post:
        post_path = POSTS_DIR / f"{selected_post}.md"
        metadata_path = post_path.with_suffix('.json')

        # Check if the post exists
        if not post_path.exists() or not metadata_path.exists():
            st.error("❌ The selected post or its metadata does not exist.")
            return

        # Load existing content and metadata
        content = get_post_content(f"{selected_post}.md")
        with open(metadata_path, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
            scheduled_time = datetime.datetime.fromisoformat(metadata['scheduled_time'])

        # Display current values
        title = st.text_input("🖊️ Post Title", value=selected_post.replace('_', ' ').title())
        content = st.text_area("📝 Content", value=content, height=300)
        scheduled_date = st.date_input("📅 Schedule Date", value=scheduled_time.date())
        scheduled_time_input = st.time_input("⏰ Schedule Time", value=scheduled_time.time())

        # Convert to user's local timezone (Pacific Time)
        local_tz = pytz.timezone("America/Los_Angeles")  # Change this to the user's local timezone
        scheduled_datetime = datetime.datetime.combine(scheduled_date, scheduled_time_input)
        scheduled_datetime = local_tz.localize(scheduled_datetime)

        if st.button("📤 Update Post"):
            # Save the updated markdown file
            with open(post_path, 'w', encoding='utf-8') as file:
                file.write(content)

            # Save updated scheduling metadata
            with open(metadata_path, 'w', encoding='utf-8') as file:
                json.dump({"scheduled_time": scheduled_datetime.isoformat()}, file)

            st.markdown(f'<div class="success-message">✅ Updated post: **{title}** scheduled for {scheduled_datetime}</div>', unsafe_allow_html=True)
            st.experimental_rerun()

def delete_blog_posts():
    if not st.session_state.logged_in:
        st.warning("⚠️ You must be logged in to delete posts.")
        return

    st.header("🗑️ Delete Blog Posts")
    posts = list_posts()
    selected_posts = st.multiselect("Select posts to delete", posts)  # Only show post names
    confirm_delete = st.checkbox("⚠️ Confirm Deletion")

    if st.button("Delete Selected") and confirm_delete:
        for post in selected_posts:
            delete_post(post)
        st.markdown('<div class="success-message">✅ Deleted successfully</div>', unsafe_allow_html=True)
        st.experimental_rerun()

def view_scheduled_posts():
    if not st.session_state.logged_in:
        st.warning("⚠️ You must be logged in to view scheduled posts.")
        return

    st.header("📅 Scheduled Posts")
    # Get current time in user's local timezone (Pacific Time)
    local_tz = pytz.timezone("America/Los_Angeles")  # Change this to the user's local timezone
    now = datetime.datetime.now(local_tz)
    scheduled_posts = []

    for post_path in POSTS_DIR.glob('*.json'):
        with open(post_path, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
            scheduled_time = datetime.datetime.fromisoformat(metadata['scheduled_time']).astimezone(local_tz)
            if scheduled_time > now:
                scheduled_posts.append((post_path.stem, scheduled_time))

    if not scheduled_posts:
        st.info("No scheduled posts available.")
        return

    for post_title, scheduled_time in scheduled_posts:
        st.markdown(f"**Post Title:** {post_title.replace('_', ' ').title()}")
        st.markdown(f"**Scheduled for:** {scheduled_time.strftime('%Y-%m-%d %H:%M')}")
        st.markdown("---")

# ===========================
# 9. Main Functionality
# ===========================

def main():
    st.sidebar.title("📂 Blog Management")
    page = st.sidebar.radio("🛠️ Choose an option", ["View Posts", "View Scheduled Posts", "Create Post", "Edit Scheduled Post", "Delete Post"])
    
    if page == "View Posts":
        view_blog_posts()
    elif page == "View Scheduled Posts":
        view_scheduled_posts()  # New option to view scheduled posts
    elif page == "Edit Scheduled Post":
        edit_scheduled_post()  # New option to edit scheduled posts
    elif page in ["Create Post", "Delete Post"]:
        login()
        if st.session_state.logged_in:
            create_blog_post() if page == "Create Post" else delete_blog_posts()

if __name__ == "__main__":
    main()

# ===========================
# 10. Footer Section
# ===========================

st.markdown('''
    <div class="footer">
        &copy; 2024 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
