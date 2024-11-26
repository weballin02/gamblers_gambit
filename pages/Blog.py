import os
import streamlit as st
from pathlib import Path
from datetime import datetime
from PIL import Image

# Define directories
POSTS_DIR = Path('posts')
IMAGES_DIR = POSTS_DIR / 'images'
TRASH_DIR = Path('trash')

# Ensure directories exist
POSTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TRASH_DIR.mkdir(parents=True, exist_ok=True)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Streamlit Blog Manager",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Authentication (Basic Example)
from streamlit import session_state as state

def login():
    if 'logged_in' not in state:
        state.logged_in = False
    
    if not state.logged_in:
        st.sidebar.header("🔒 Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username == "admin" and password == "password":  # Replace with secure authentication
                state.logged_in = True
                st.sidebar.success("✅ Logged in successfully!")
            else:
                st.sidebar.error("❌ Invalid credentials")

# Helper Functions
def list_posts():
    return sorted([f.name for f in POSTS_DIR.glob('*.md')], reverse=True)

def delete_post(post_name):
    post_path = POSTS_DIR / post_name
    if post_path.exists():
        os.remove(post_path)
        return True
    return False

def move_to_trash(post_name):
    post_path = POSTS_DIR / post_name
    trash_path = TRASH_DIR / post_name
    if post_path.exists():
        post_path.rename(trash_path)
        return True
    return False

# Streamlit Interface Functions
def view_blog_posts():
    st.header("📖 Explore Gambler's Gambit")
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

    # Display Posts in a Card Layout
    st.markdown("""
        <style>
        .post-card {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            color: #FFF;
        }
        .post-title {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .post-meta {
            font-size: 0.9em;
            color: #AAAAAA;
            margin-bottom: 15px;
        }
        .post-content {
            font-size: 1em;
            line-height: 1.6;
        }
        .post-image {
            max-width: 100%;
            height: auto;
            margin-bottom: 15px;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .read-more {
            display: inline-block;
            margin-top: 10px;
            font-size: 1em;
            color: #007BFF;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .read-more:hover {
            color: #0056b3;
        }
        </style>
    """, unsafe_allow_html=True)

    for post in filtered_posts:
        post_title = post.replace('.md', '').replace('_', ' ').title()
        post_file = POSTS_DIR / post
        image_file = IMAGES_DIR / f"{post.replace('.md', '.jpg')}"
        
        # Read and process post content
        with open(post_file, 'r') as file:
            content = file.read()
            content_preview = content[:200] + "..." if len(content) > 200 else content

        # Get publication date from file metadata
        pub_date = datetime.fromtimestamp(post_file.stat().st_mtime).strftime("%B %d, %Y at %H:%M")

        # Display post in a card
        st.markdown(f"""
            <div class="post-card">
                {"<img src='" + str(image_file) + "' class='post-image'>" if image_file.exists() else ""}
                <div class="post-title">{post_title}</div>
                <div class="post-meta">Published on: {pub_date}</div>
                <div class="post-content">{content_preview}</div>
                <a href="#" class="read-more" onClick="document.getElementById('{post}').style.display='block'; return false;">Read More</a>
            </div>
            <div id="{post}" style="display:none; margin-top: 20px;">
                <div class="post-card">
                    <div class="post-title">{post_title}</div>
                    <div class="post-meta">Published on: {pub_date}</div>
                    <div class="post-content">{content}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def create_blog_post():
    st.header("📝 Create a New Blog Post")
    with st.form(key='create_post_form'):
        title = st.text_input("🖊️ Post Title", placeholder="Enter the title of your post")
        content = st.text_area("📝 Content", height=300, placeholder="Write your post content here...")
        image = st.file_uploader("📷 Upload a Feature Image", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("📤 Publish")
        
        if submitted:
            if title and content:
                filename = f"{title.replace(' ', '_').lower()}.md"
                filepath = POSTS_DIR / filename
                if filepath.exists():
                    st.error("❌ A post with this title already exists. Please choose a different title.")
                else:
                    # Save content
                    with open(filepath, 'w') as file:
                        file.write(content)
                    # Save image
                    if image:
                        image_path = IMAGES_DIR / f"{filename.replace('.md', '.jpg')}"
                        with open(image_path, "wb") as img_file:
                            img_file.write(image.read())
                    st.success(f"✅ Published post: **{title}**")
                    st.experimental_rerun()
            else:
                st.warning("⚠️ Please provide both a title and content for the post.")

def delete_blog_posts():
    st.header("🗑️ Delete Blog Posts")
    posts = list_posts()
    if not posts:
        st.info("No blog posts available to delete.")
        return
    
    confirm_delete = st.checkbox("⚠️ I understand that deleting a post is irreversible.")
    selected_posts = st.multiselect("Select posts to delete", posts)
    
    if st.button("🗑️ Move Selected Posts to Trash") and confirm_delete:
        for post in selected_posts:
            move_to_trash(post)
            st.success(f"✅ Moved to trash: **{post.replace('.md', '').replace('_', ' ').title()}**")
        st.experimental_rerun()

# Main Function
def main():
    st.title("📝 Streamlit Blog Manager")
    st.sidebar.title("📂 Blog Management")
    page = st.sidebar.radio("🛠️ Choose an option", ["View Posts", "Create Post", "Delete Post"])
    
    if page == "View Posts":
        view_blog_posts()
    elif page in ["Create Post", "Delete Post"]:
        login()
        if state.logged_in:
            if page == "Create Post":
                create_blog_post()
            elif page == "Delete Post":
                delete_blog_posts()
        else:
            st.warning("🔒 Please log in to access this feature.")

if __name__ == "__main__":
    main()
