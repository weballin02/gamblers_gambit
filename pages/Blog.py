# blog.py

# ===========================
# 1. Import Libraries
# ===========================

import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
from pathlib import Path
from PIL import Image
import datetime
import pytz  # Import pytz for timezone handling
from io import BytesIO
import base64
import json  # For handling JSON data
import fitz  # PyMuPDF for PDF processing
import html2text  # For converting HTML to Markdown

# ===========================
# 2. Initialize Firebase
# ===========================

# Initialize Firebase only if not already initialized
if not firebase_admin._apps:
    cred_info = {
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
    }
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'your-project-id.appspot.com'  # Replace with your actual storage bucket
    })

# Firestore client
db = firestore.client()

# Storage bucket
bucket = storage.bucket()

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
            background: var(--accent-color-purple);
            color: var(--primary-text-color);
            border: none;
            border-radius: 5px;
            padding: 5px 10px;
            cursor: pointer;
            transition: background 0.3s ease;
            font-family: 'Montserrat', sans-serif;
        }

        .read-more-button:hover {
            background: var(--accent-color-teal);
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

def create_firestore_post(title, content, scheduled_time, image_url=None):
    try:
        doc_ref = db.collection('posts').document()
        doc_ref.set({
            'title': title,
            'content': content,
            'scheduled_time': scheduled_time,
            'image_url': image_url,
            'created_at': datetime.datetime.utcnow()
        })
        st.markdown(
            f'<div class="success-message">✅ Published post: **{title}** scheduled for {scheduled_time.astimezone(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M")}</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"❌ Failed to create post: {e}")

def list_firestore_posts():
    now = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
    posts_ref = db.collection('posts').where('scheduled_time', '<=', now).order_by('scheduled_time', direction=firestore.Query.DESCENDING)
    try:
        docs = posts_ref.stream()
        posts = []
        for doc in docs:
            post = doc.to_dict()
            post['id'] = doc.id
            posts.append(post)
        return posts
    except Exception as e:
        st.error(f"❌ Failed to fetch posts: {e}")
        return []

def delete_firestore_post(post_id):
    try:
        post_ref = db.collection('posts').document(post_id)
        post = post_ref.get()
        if post.exists:
            # Delete associated image from Storage if exists
            image_url = post.to_dict().get('image_url')
            if image_url:
                # Extract the blob path from the URL
                blob_path = image_url.split('.com/')[1]
                blob = bucket.blob(blob_path)
                if blob.exists():
                    blob.delete()
            # Delete the Firestore document
            post_ref.delete()
            st.success("✅ Post deleted successfully.")
        else:
            st.error("❌ Post does not exist.")
    except Exception as e:
        st.error(f"❌ Failed to delete post: {e}")

def update_firestore_post(post_id, title, content, scheduled_time, image_file=None):
    try:
        post_ref = db.collection('posts').document(post_id)
        post = post_ref.get()
        if not post.exists:
            st.error("❌ Post does not exist.")
            return

        update_data = {
            'title': title,
            'content': content,
            'scheduled_time': scheduled_time
        }

        if image_file:
            # Delete old image if exists
            old_image_url = post.to_dict().get('image_url')
            if old_image_url:
                old_blob_path = old_image_url.split('.com/')[1]
                old_blob = bucket.blob(old_blob_path)
                if old_blob.exists():
                    old_blob.delete()
            # Upload new image
            blob = bucket.blob(f'images/{uuid.uuid4()}_{image_file.name}')
            blob.upload_from_file(image_file, content_type=image_file.type)
            blob.make_public()
            new_image_url = blob.public_url
            update_data['image_url'] = new_image_url

        post_ref.update(update_data)
        st.success(f"✅ Updated post: **{title}** scheduled for {scheduled_time.astimezone(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        st.error(f"❌ Failed to update post: {e}")

def get_post_content(post_id):
    try:
        post_ref = db.collection('posts').document(post_id)
        post = post_ref.get()
        if post.exists:
            return post.to_dict().get('content', "Post content not found.")
        else:
            return "Post not found."
    except Exception as e:
        st.error(f"❌ Failed to fetch post content: {e}")
        return "Error fetching content."

def display_full_post(post_id):
    try:
        post_ref = db.collection('posts').document(post_id)
        post = post_ref.get()
        if not post.exists:
            st.error("❌ Post not found.")
            return

        post_data = post.to_dict()
        st.button("🔙 Back to Posts", on_click=lambda: st.session_state.update(selected_post=None))
        st.header(post_data['title'])
        st.markdown(post_data['content'])

        if post_data.get('image_url'):
            st.image(post_data['image_url'], caption="Thumbnail", use_column_width=True)
    except Exception as e:
        st.error(f"❌ Failed to display post: {e}")

# ===========================
# 8. Streamlit Interface Functions
# ===========================

def view_blog_posts():
    st.header("📖 Explore The Gambit")

    if st.session_state.selected_post:
        display_full_post(st.session_state.selected_post)
        return

    posts = list_firestore_posts()
    if not posts:
        st.info("No blog posts available.")
        return

    search_query = st.text_input("🔍 Search Posts", "")
    if search_query:
        posts = [post for post in posts if search_query.lower() in post['title'].lower()]

    if not posts:
        st.warning("No posts match your search.")
        return

    for post in posts:
        post_title = post["title"]
        content_preview = post["content"][:200] + "..."
        image_tag = ""

        if post["image_url"]:
            image_tag = f'<img src="{post["image_url"]}" width="150" height="100" />'
        else:
            image_tag = '<div style="width:150px; height:100px; background-color:#2C3E50; border-radius:5px;"></div>'

        pub_date = post['scheduled_time'].astimezone(pytz.timezone("America/Los_Angeles")).strftime('%Y-%m-%d %H:%M')
        read_more_key = f"read_more_{post['id']}"

        # Card layout for each post
        st.markdown(f"""
            <div class="post-card">
                <div class="post-image">
                    {image_tag}
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
            st.session_state.selected_post = post["id"]
            st.experimental_rerun()

def create_blog_post():
    st.header("📝 Create a New Blog Post")
    post_type = st.selectbox("Select Post Type", ["Manual Entry", "Upload PDF/HTML"])

    # Initialize title and content
    title = ""
    content = ""

    if post_type == "Manual Entry":
        title = st.text_input("🖊️ Post Title")
        content = st.text_area("📝 Content", height=300)
    elif post_type == "Upload PDF/HTML":
        uploaded_file = st.file_uploader("📂 Upload PDF or HTML File", type=["pdf", "html"])
        title = st.text_input("🖊️ Post Title (Optional)", placeholder="Enter the title of your post (optional)")

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

    # Convert to UTC for consistency
    local_tz = pytz.timezone("America/Los_Angeles")  # Adjust if needed
    localized_datetime = local_tz.localize(scheduled_datetime)
    utc_datetime = localized_datetime.astimezone(pytz.UTC)

    if st.button("📤 Publish"):
        if not title:
            st.warning("⚠️ Please provide a title for the post.")
            return
        if not content:
            st.warning("⚠️ Please provide content for the post.")
            return

        # Optional: Validate scheduled_time is in the future
        if utc_datetime < datetime.datetime.utcnow().replace(tzinfo=pytz.UTC):
            st.warning("⚠️ Scheduled time must be in the future.")
            return

        # Handle Image Upload
        image_url = ""
        if image:
            try:
                # Compress the image
                img = Image.open(image)
                img_io = BytesIO()
                img.save(img_io, format='PNG', optimize=True, quality=70)
                img_io.seek(0)

                # Upload to Firebase Storage
                blob = bucket.blob(f'images/{uuid.uuid4()}_{image.name}')
                blob.upload_from_file(img_io, content_type=image.type)
                blob.make_public()
                image_url = blob.public_url
            except Exception as e:
                st.error(f"❌ Failed to upload image: {e}")
                return

        # Create Firestore document
        create_firestore_post(title, content, utc_datetime, image_url)
        st.experimental_rerun()

def edit_scheduled_post():
    if not st.session_state.logged_in:
        st.warning("⚠️ You must be logged in to edit scheduled posts.")
        return

    st.header("✏️ Edit Scheduled Post")
    # Get all scheduled posts (scheduled_time in the future)
    now = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
    scheduled_posts_ref = db.collection('posts').where('scheduled_time', '>', now)
    try:
        docs = scheduled_posts_ref.stream()
        scheduled_posts = []
        for doc in docs:
            post = doc.to_dict()
            post['id'] = doc.id
            scheduled_posts.append(post)
    except Exception as e:
        st.error(f"❌ Failed to fetch scheduled posts: {e}")
        return

    if not scheduled_posts:
        st.info("No scheduled posts available.")
        return

    # Create a selection list
    post_options = {f"{post['title']} (Scheduled for {post['scheduled_time'].astimezone(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M')})": post['id'] for post in scheduled_posts}
    selected_post_display = st.selectbox("Select a post to edit", list(post_options.keys()))
    selected_post_id = post_options.get(selected_post_display)

    if selected_post_id:
        post_ref = db.collection('posts').document(selected_post_id)
        post = post_ref.get()
        if not post.exists:
            st.error("❌ Selected post does not exist.")
            return

        post_data = post.to_dict()
        title = st.text_input("🖊️ Post Title", value=post_data.get('title', ''))
        content = st.text_area("📝 Content", value=post_data.get('content', ''), height=300)
        scheduled_datetime = post_data.get('scheduled_time').astimezone(pytz.timezone("America/Los_Angeles"))
        scheduled_date = st.date_input("📅 Schedule Date", value=scheduled_datetime.date())
        scheduled_time = st.time_input("⏰ Schedule Time", value=scheduled_datetime.time())

        # Combine date and time
        new_scheduled_datetime = datetime.datetime.combine(scheduled_date, scheduled_time)
        localized_new_scheduled_datetime = local_tz.localize(new_scheduled_datetime)
        utc_new_scheduled_datetime = localized_new_scheduled_datetime.astimezone(pytz.UTC)

        image = st.file_uploader("🖼️ Upload New Thumbnail Image (Optional)", type=["png", "jpg", "jpeg"])

        if st.button("📤 Update Post"):
            if not title:
                st.warning("⚠️ Please provide a title for the post.")
                return
            if not content:
                st.warning("⚠️ Please provide content for the post.")
                return

            # Handle Image Upload if a new image is provided
            image_url = post_data.get('image_url', "")
            if image:
                try:
                    # Compress the image
                    img = Image.open(image)
                    img_io = BytesIO()
                    img.save(img_io, format='PNG', optimize=True, quality=70)
                    img_io.seek(0)

                    # Upload to Firebase Storage
                    blob = bucket.blob(f'images/{uuid.uuid4()}_{image.name}')
                    blob.upload_from_file(img_io, content_type=image.type)
                    blob.make_public()
                    image_url = blob.public_url

                    # Delete old image if exists
                    old_image_url = post_data.get('image_url')
                    if old_image_url:
                        old_blob_path = old_image_url.split('.com/')[1]
                        old_blob = bucket.blob(old_blob_path)
                        if old_blob.exists():
                            old_blob.delete()
                except Exception as e:
                    st.error(f"❌ Failed to upload new image: {e}")
                    return

            # Update Firestore document
            update_firestore_post(selected_post_id, title, content, utc_new_scheduled_datetime, image_file=image if image else None)
            st.experimental_rerun()

def delete_blog_posts():
    if not st.session_state.logged_in:
        st.warning("⚠️ You must be logged in to delete posts.")
        return

    st.header("🗑️ Delete Blog Posts")
    posts = list_firestore_posts()
    if not posts:
        st.info("No blog posts available.")
        return

    # Create selection list
    post_options = {f"{post['title']} (Published on {post['scheduled_time'].astimezone(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M')})": post["id"] for post in posts}
    selected_posts = st.multiselect("Select posts to delete", list(post_options.keys()))
    confirm_delete = st.checkbox("⚠️ Confirm Deletion")

    if st.button("Delete Selected") and confirm_delete:
        for post_display in selected_posts:
            post_id = post_options.get(post_display)
            if post_id:
                delete_firestore_post(post_id)
        st.experimental_rerun()

def view_scheduled_posts():
    if not st.session_state.logged_in:
        st.warning("⚠️ You must be logged in to view scheduled posts.")
        return

    st.header("📅 Scheduled Posts")
    now = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
    scheduled_posts_ref = db.collection('posts').where('scheduled_time', '>', now).order_by('scheduled_time', direction=firestore.Query.ASCENDING)
    try:
        docs = scheduled_posts_ref.stream()
        scheduled_posts = []
        for doc in docs:
            post = doc.to_dict()
            post['id'] = doc.id
            scheduled_posts.append(post)
    except Exception as e:
        st.error(f"❌ Failed to fetch scheduled posts: {e}")
        return

    if not scheduled_posts:
        st.info("No scheduled posts available.")
        return

    for post in scheduled_posts:
        post_title = post["title"]
        scheduled_time = post["scheduled_time"].astimezone(pytz.timezone("America/Los_Angeles")).strftime('%Y-%m-%d %H:%M')
        st.markdown(f"**Post Title:** {post_title}")
        st.markdown(f"**Scheduled for:** {scheduled_time}")
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
        view_scheduled_posts()
    elif page == "Edit Scheduled Post":
        edit_scheduled_post()
    elif page in ["Create Post", "Delete Post"]:
        login()
        if st.session_state.logged_in:
            if page == "Create Post":
                create_blog_post()
            else:
                delete_blog_posts()

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
