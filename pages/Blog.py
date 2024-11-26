import os
import streamlit as st
from pathlib import Path
from PIL import Image
import shutil
import datetime
from io import BytesIO
import base64

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
    if not POSTS_DIR.exists():
        POSTS_DIR.mkdir(parents=True)
    posts = sorted([f.name for f in POSTS_DIR.glob('*.md')], reverse=True)
    return posts

def delete_post(post_name):
    post_path = POSTS_DIR / post_name
    image_path = IMAGES_DIR / f"{post_path.stem}.png"  # Assuming PNG; adjust as needed
    if post_path.exists():
        os.remove(post_path)
        if image_path.exists():
            os.remove(image_path)
        return True
    else:
        return False

def move_to_trash(post_name):
    if not TRASH_DIR.exists():
        TRASH_DIR.mkdir(parents=True)
    post_path = POSTS_DIR / post_name
    trash_post_path = TRASH_DIR / post_name
    image_path = IMAGES_DIR / f"{post_path.stem}.png"  # Assuming PNG; adjust as needed
    trash_image_path = TRASH_DIR / f"{post_path.stem}.png"
    
    if post_path.exists():
        post_path.rename(trash_post_path)
        if image_path.exists():
            image_path.rename(trash_image_path)
        return True
    else:
        return False

def get_post_content(post_name):
    post_file = POSTS_DIR / post_name
    if post_file.exists():
        with open(post_file, 'r') as file:
            content = file.read()
        return content
    return "Post content not found."

# Streamlit Interface Functions
def view_blog_posts():
    st.header("📖 Explore Our Blog")
    
    # Check if a post is selected for detailed view
    if 'selected_post' in state and state.selected_post:
        display_full_post(state.selected_post)
        return
    
    posts = list_posts()
    if not posts:
        st.info("No blog posts available.")
        return
    
    # Search Functionality
    search_query = st.text_input("🔍 Search Posts", "")
    if search_query:
        filtered_posts = [post for post in posts if search_query.lower() in post.lower()]
    else:
        filtered_posts = posts
    
    if not filtered_posts:
        st.warning("No posts match your search.")
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
    for post in filtered_posts:
        post_title = post.replace('.md', '').replace('_', ' ').title()
        post_file = POSTS_DIR / post
        
        # Read and process post content
        content = get_post_content(post)
        content_preview = content[:200] + "..." if len(content) > 200 else content
        
        # Check for associated image
        image_file = IMAGES_DIR / f"{post_file.stem}.png"  # Assuming PNG; adjust as needed
        if image_file.exists():
            image_path = image_file
        else:
            image_path = None  # Placeholder or default image can be set here

        # Convert image to base64 for embedding
        if image_path:
            img = Image.open(image_path)
            img.thumbnail((150, 100))
            buf = BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            encoded = base64.b64encode(img_bytes).decode()
            img_html = f'<img src="data:image/png;base64,{encoded}" class="thumbnail"/>'
        else:
            img_html = '<div style="width:150px; height:100px; background-color:#cccccc; border-radius:5px; margin-right:20px;"></div>'

        # Format publication date
        pub_date = datetime.datetime.fromtimestamp(post_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')

        # Unique key for each "Read More" button
        read_more_key = f"read_more_{post}"

        # Display post in a card with "Read More" button
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
            
            # Capture the "Read More" click using Streamlit's experimental components
            if st.button("Read More", key=read_more_key):
                state.selected_post = post

    st.markdown("---")
    st.markdown("""
        <div style="text-align:center; margin-top:20px;">
            Want to contribute or learn more? Reach out to us at <a href="mailto:info@yourblog.com">info@yourblog.com</a>.
        </div>
    """, unsafe_allow_html=True)

def display_full_post(post_name):
    st.subheader("🔙 Back to Posts")
    if st.button("← Back"):
        state.selected_post = None
        st.experimental_rerun()
    
    post_title = post_name.replace('.md', '').replace('_', ' ').title()
    post_file = POSTS_DIR / post_name
    content = get_post_content(post_name)
    
    # Display the post title
    st.title(post_title)
    
    # Display publication date
    pub_date = datetime.datetime.fromtimestamp(post_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    st.markdown(f"**Published on:** {pub_date}")
    
    # Display the image if exists
    image_file = IMAGES_DIR / f"{post_file.stem}.png"
    if image_file.exists():
        st.image(str(image_file), use_column_width=True)
    
    # Display the full content
    st.markdown(content)

def create_blog_post():
    st.header("📝 Create a New Blog Post")
    with st.form(key='create_post_form'):
        title = st.text_input("🖊️ Post Title", placeholder="Enter the title of your post")
        content = st.text_area("📝 Content", height=300, placeholder="Write your post content here...")
        image = st.file_uploader("🖼️ Upload Thumbnail Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        submitted = st.form_submit_button("📤 Publish")
        
        if submitted:
            if title and content:
                filename = f"{title.replace(' ', '_').lower()}.md"
                filepath = POSTS_DIR / filename
                image_filename = f"{filepath.stem}.png"  # Saving all images as PNG for consistency
                image_path = IMAGES_DIR / image_filename

                if filepath.exists():
                    st.error("❌ A post with this title already exists. Please choose a different title.")
                else:
                    # Save the markdown file
                    with open(filepath, 'w') as file:
                        file.write(content)
                    
                    # Save the uploaded image if provided
                    if image:
                        try:
                            img = Image.open(image)
                            img.save(image_path, format="PNG")
                            st.success(f"✅ Published post with image: **{title}**")
                        except Exception as e:
                            st.error(f"❌ Failed to save image: {e}")
                            # Optionally, you might want to delete the markdown file if image saving fails
                    else:
                        st.success(f"✅ Published post: **{title}** (No image uploaded)")
                    
                    st.experimental_rerun()
            else:
                st.warning("⚠️ Please provide both a title and content for the post.")

def delete_blog_posts():
    st.header("🗑️ Delete Blog Posts")
    
    posts = list_posts()
    if not posts:
        st.info("No blog posts available to delete.")
        return
    
    # Confirmation Checkbox
    confirm_delete = st.checkbox("⚠️ I understand that deleting a post is irreversible.")
    
    # Display posts with delete options
    selected_posts = st.multiselect("Select posts to delete", posts)
    
    if selected_posts:
        cols = st.columns([1, 5])
        with cols[0]:
            pass  # Spacer
        with cols[1]:
            st.markdown("### Selected Posts for Deletion")
            for post in selected_posts:
                st.write(f"- {post.replace('.md', '').replace('_', ' ').title()}")
    
    # Delete Button
    if st.button("🗑️ Move Selected Posts to Trash") and confirm_delete:
        if selected_posts:
            for post in selected_posts:
                success = move_to_trash(post)
                if success:
                    st.success(f"✅ Moved to trash: **{post.replace('.md', '').replace('_', ' ').title()}**")
                else:
                    st.error(f"❌ Failed to move: **{post}**")
            st.experimental_rerun()
        else:
            st.warning("⚠️ No posts selected for deletion.")
    elif st.button("🗑️ Move Selected Posts to Trash") and not confirm_delete:
        st.warning("⚠️ Please confirm deletion by checking the box above.")

# Additional Features
def display_header():
    st.title("📝 Streamlit Blog Manager")
    st.markdown("""
    Welcome to the **Streamlit Blog Manager**! Use the sidebar to navigate between viewing, creating, and deleting blog posts.
    """)

# Main Function
def main():
    display_header()
    
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
