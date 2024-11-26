import os
import streamlit as st
from pathlib import Path
from PIL import Image
import shutil
import datetime
from io import BytesIO
import base64
import mimetypes

# Define directories
POSTS_DIR = Path('posts')
TRASH_DIR = Path('trash')
IMAGES_DIR = Path('images')    # Directory for images
PDFS_DIR = Path('pdfs')        # Directory for PDFs
HTML_DIR = Path('htmls')        # Directory for HTML files

# Ensure directories exist
for directory in [POSTS_DIR, TRASH_DIR, IMAGES_DIR, PDFS_DIR, HTML_DIR]:
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
    pdf_path = PDFS_DIR / f"{post_path.stem}.pdf"
    html_path = HTML_DIR / f"{post_path.stem}.html"
    if post_path.exists():
        os.remove(post_path)
        if image_path.exists():
            os.remove(image_path)
        if pdf_path.exists():
            os.remove(pdf_path)
        if html_path.exists():
            os.remove(html_path)
        return True
    else:
        return False

def move_to_trash(post_name):
    if not TRASH_DIR.exists():
        TRASH_DIR.mkdir(parents=True)
    post_path = POSTS_DIR / post_name
    trash_post_path = TRASH_DIR / post_name
    image_path = IMAGES_DIR / f"{post_path.stem}.png"
    trash_image_path = TRASH_DIR / f"{post_path.stem}.png"
    pdf_path = PDFS_DIR / f"{post_path.stem}.pdf"
    trash_pdf_path = TRASH_DIR / f"{post_path.stem}.pdf"
    html_path = HTML_DIR / f"{post_path.stem}.html"
    trash_html_path = TRASH_DIR / f"{post_path.stem}.html"
    
    if post_path.exists():
        post_path.rename(trash_post_path)
        if image_path.exists():
            image_path.rename(trash_image_path)
        if pdf_path.exists():
            pdf_path.rename(trash_pdf_path)
        if html_path.exists():
            html_path.rename(trash_html_path)
        return True
    else:
        return False

def get_post_content(post_name):
    post_file = POSTS_DIR / post_name
    if post_file.exists():
        with open(post_file, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    return "Post content not found."

# File Serving Functions
def serve_pdf(pdf_path):
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
            # Encode the PDF data in base64
            encoded = base64.b64encode(pdf_data).decode()
            # Create an iframe to embed the PDF
            pdf_display = f'<iframe src="data:application/pdf;base64,{encoded}" width="700" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error("❌ PDF file not found.")

def serve_html(html_path):
    if html_path.exists():
        with open(html_path, "r", encoding='utf-8') as f:
            html_content = f.read()
            # Use Streamlit's components to render HTML
            import streamlit.components.v1 as components
            components.html(html_content, height=600, scrolling=True)
    else:
        st.error("❌ HTML file not found.")

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
        .additional-files {
            margin-top: 10px;
            font-size: 0.9em;
            color: #007BFF;
        }
        .additional-files a {
            margin-right: 10px;
            text-decoration: none;
            color: #007BFF;
        }
        .additional-files a:hover {
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
        image_file = IMAGES_DIR / f"{post_file.stem}.png"
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

        # Check for associated PDF and HTML
        pdf_file = PDFS_DIR / f"{post_file.stem}.pdf"
        html_file = HTML_DIR / f"{post_file.stem}.html"

        additional_files_html = ""
        if pdf_file.exists():
            # Since Streamlit cannot directly link to file paths, we'll provide download and view options
            additional_files_html += f'<a href="#" onclick="window.open(\'#\', \'_blank\')">📄 PDF</a>'
        if html_file.exists():
            additional_files_html += f'<a href="#" onclick="window.open(\'#\', \'_blank\')">🌐 HTML</a>'
        if additional_files_html:
            additional_files_html = f'<div class="additional-files">{additional_files_html}</div>'

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
                        {additional_files_html}
                        <a href="#" class="read-more" id="{read_more_key}">Read More</a>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Capture the "Read More" click using Streamlit's experimental components
            if st.button("Read More", key=read_more_key):
                state.selected_post = post

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
    st.markdown(content, unsafe_allow_html=True)
    
    # Display the PDF if exists
    pdf_file = PDFS_DIR / f"{post_file.stem}.pdf"
    if pdf_file.exists():
        st.markdown("### 📄 PDF")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download PDF",
                data=pdf_file.read_bytes(),
                file_name=pdf_file.name,
                mime="application/pdf"
            )
        with col2:
            if st.button("📖 View PDF"):
                serve_pdf(pdf_file)
    
    # Display the HTML if exists
    html_file = HTML_DIR / f"{post_file.stem}.html"
    if html_file.exists():
        st.markdown("### 🌐 HTML Content")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download HTML",
                data=html_file.read_bytes(),
                file_name=html_file.name,
                mime="text/html"
            )
        with col2:
            if st.button("🔍 View HTML"):
                serve_html(html_file)

def create_blog_post():
    st.header("📝 Create a New Blog Post")
    with st.form(key='create_post_form'):
        title = st.text_input("🖊️ Post Title", placeholder="Enter the title of your post")
        content = st.text_area("📝 Content", height=300, placeholder="Write your post content here...")
        image = st.file_uploader("🖼️ Upload Thumbnail Image (Optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        pdf_file = st.file_uploader("📚 Upload PDF File (Optional)", type=["pdf"], accept_multiple_files=False)
        html_file = st.file_uploader("🌐 Upload HTML File (Optional)", type=["html"], accept_multiple_files=False)
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
                    try:
                        with open(filepath, 'w', encoding='utf-8') as file:
                            file.write(content)
                        st.success(f"✅ Published post: **{title}**")
                    except Exception as e:
                        st.error(f"❌ Failed to save Markdown file: {e}")
                        return

                    # Save the uploaded image if provided
                    if image:
                        image_extension = Path(image.name).suffix.lower()
                        if image_extension not in ['.png', '.jpg', '.jpeg']:
                            st.error("❌ Unsupported image format. Please upload PNG, JPG, or JPEG files.")
                        else:
                            try:
                                img = Image.open(image)
                                img.save(image_path)
                                st.success(f"✅ Uploaded image: **{image_filename}**")
                            except Exception as e:
                                st.error(f"❌ Failed to save image: {e}")

                    # Save the PDF file if uploaded
                    if pdf_file:
                        pdf_extension = Path(pdf_file.name).suffix.lower()
                        if pdf_extension != '.pdf':
                            st.error("❌ Unsupported PDF format. Please upload a PDF file.")
                        else:
                            pdf_filename = f"{filepath.stem}.pdf"
                            pdf_path = PDFS_DIR / pdf_filename
                            try:
                                with open(pdf_path, 'wb') as f:
                                    f.write(pdf_file.getbuffer())
                                st.success(f"✅ Uploaded PDF: **{pdf_filename}**")
                            except Exception as e:
                                st.error(f"❌ Failed to save PDF file: {e}")

                    # Save the HTML file if uploaded
                    if html_file:
                        html_extension = Path(html_file.name).suffix.lower()
                        if html_extension != '.html':
                            st.error("❌ Unsupported HTML format. Please upload an HTML file.")
                        else:
                            html_filename = f"{filepath.stem}.html"
                            html_path = HTML_DIR / html_filename
                            try:
                                with open(html_path, 'wb') as f:
                                    f.write(html_file.getbuffer())
                                st.success(f"✅ Uploaded HTML: **{html_filename}**")
                            except Exception as e:
                                st.error(f"❌ Failed to save HTML file: {e}")
                    
                    st.experimental_rerun()
            else:
                st.warning("⚠️ Please provide both a title and content for the post.")

def import_blog_post():
    st.header("📥 Import Existing Blog Post")
    with st.form(key='import_post_form'):
        md_file = st.file_uploader("📄 Upload Markdown File", type=["md"], accept_multiple_files=False)
        image_file = st.file_uploader("🖼️ Upload Associated Image (Optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        pdf_file = st.file_uploader("📚 Upload PDF File (Optional)", type=["pdf"], accept_multiple_files=False)
        html_file = st.file_uploader("🌐 Upload HTML File (Optional)", type=["html"], accept_multiple_files=False)
        submitted = st.form_submit_button("📥 Import Post")
        
        if submitted:
            if not md_file:
                st.error("❌ Please upload a Markdown (.md) file.")
            else:
                # Extract filename and ensure it's unique
                md_filename = md_file.name
                if not md_filename.lower().endswith('.md'):
                    st.error("❌ The uploaded file is not a Markdown (.md) file.")
                else:
                    # Sanitize and create a valid filename
                    sanitized_title = Path(md_filename).stem.replace(' ', '_').lower()
                    new_md_filename = f"{sanitized_title}.md"
                    new_md_filepath = POSTS_DIR / new_md_filename

                    if new_md_filepath.exists():
                        st.error("❌ A post with this title already exists. Please choose a different file or rename it.")
                    else:
                        # Save the Markdown file
                        try:
                            with open(new_md_filepath, 'wb') as f:
                                f.write(md_file.getbuffer())
                            st.success(f"✅ Imported Markdown file: **{new_md_filename}**")
                        except Exception as e:
                            st.error(f"❌ Failed to save Markdown file: {e}")
                            return  # Exit if saving fails

                        # Handle the image file if uploaded
                        if image_file:
                            image_extension = Path(image_file.name).suffix.lower()
                            if image_extension not in ['.png', '.jpg', '.jpeg']:
                                st.error("❌ Unsupported image format. Please upload PNG, JPG, or JPEG files.")
                            else:
                                image_filename = f"{new_md_filepath.stem}{image_extension}"
                                image_path = IMAGES_DIR / image_filename
                                try:
                                    img = Image.open(image_file)
                                    img.save(image_path)
                                    st.success(f"✅ Imported image: **{image_filename}**")
                                except Exception as e:
                                    st.error(f"❌ Failed to save image: {e}")
                        
                        # Handle the PDF file if uploaded
                        if pdf_file:
                            pdf_extension = Path(pdf_file.name).suffix.lower()
                            if pdf_extension != '.pdf':
                                st.error("❌ Unsupported PDF format. Please upload a PDF file.")
                            else:
                                pdf_filename = f"{new_md_filepath.stem}.pdf"
                                pdf_path = PDFS_DIR / pdf_filename
                                try:
                                    with open(pdf_path, 'wb') as f:
                                        f.write(pdf_file.getbuffer())
                                    st.success(f"✅ Imported PDF: **{pdf_filename}**")
                                except Exception as e:
                                    st.error(f"❌ Failed to save PDF file: {e}")
                        
                        # Handle the HTML file if uploaded
                        if html_file:
                            html_extension = Path(html_file.name).suffix.lower()
                            if html_extension != '.html':
                                st.error("❌ Unsupported HTML format. Please upload an HTML file.")
                            else:
                                html_filename = f"{new_md_filepath.stem}.html"
                                html_path = HTML_DIR / html_filename
                                try:
                                    with open(html_path, 'wb') as f:
                                        f.write(html_file.getbuffer())
                                    st.success(f"✅ Imported HTML: **{html_filename}**")
                                except Exception as e:
                                    st.error(f"❌ Failed to save HTML file: {e}")
                        
                        st.experimental_rerun()

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

def display_header():
    st.title("📝 Streamlit Blog Manager")
    st.markdown("""
    Welcome to the **Streamlit Blog Manager**! Use the sidebar to navigate between viewing, creating, importing, and deleting blog posts.
    """)

# Main Function
def main():
    display_header()
    
    st.sidebar.title("📂 Blog Management")
    page = st.sidebar.radio("🛠️ Choose an option", ["View Posts", "Create Post", "Import Post", "Delete Post"])
    
    if page == "View Posts":
        view_blog_posts()
    elif page in ["Create Post", "Import Post", "Delete Post"]:
        login()
        if state.logged_in:
            if page == "Create Post":
                create_blog_post()
            elif page == "Import Post":
                import_blog_post()
            elif page == "Delete Post":
                delete_blog_posts()
        else:
            st.warning("🔒 Please log in to access this feature.")

if __name__ == "__main__":
    main()
