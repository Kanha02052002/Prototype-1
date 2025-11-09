import streamlit as st
import requests
import time
import os
import shutil


API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="IT Support Chatbot", page_icon="🤖", layout="centered")

# --- Custom CSS for styling ---
st.markdown("""
<style>
.chat-container {
    max-width: 700px;
    margin: auto;
}
.user-msg, .bot-msg {
    border-radius: 15px;
    padding: 10px 15px;
    margin: 8px 0;
    width: fit-content;
    max-width: 85%;
    animation: fadeIn 0.3s ease-in-out;
}
.user-msg {
    background-color: #0078ff;
    color: white;
    margin-left: auto;
}
.bot-msg {
    background-color: #2b2b2b;
    color: white;
    margin-right: auto;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
.typing {
    color: #888;
    font-style: italic;
    margin: 5px 0;
}
.scroll-container {
    height: 7vh;
    overflow-y: auto;
    padding-right: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- Session state setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "history" not in st.session_state:
    st.session_state.history = []
if "upload_dir_created" not in st.session_state:
    st.session_state.upload_dir_created = False

# Create upload directory if it doesn't exist and clear it
upload_dir = "upload"
if not st.session_state.upload_dir_created:
    os.makedirs(upload_dir, exist_ok=True)
    # Clear the upload directory when starting
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting file/directory {file_path}: {e}")
    st.session_state.upload_dir_created = True

# --- Start chat session automatically ---
if not st.session_state.session_id:
    with st.spinner("Starting chat session..."):
        resp = requests.post(f"{API_BASE}/chat/start")
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.history = [{"role": "bot", "content": data["greeting"]}]
        else:
            st.error("❌ Failed to start chat session.")
            st.stop()

# --- Title ---
st.title("💬 IT Support Chatbot")
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# --- Sidebar for file upload (appears after 4th question) ---
if len([msg for msg in st.session_state.history if msg["role"] == "user"]) >= 5:
    with st.sidebar:
        st.header("📁 Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["txt", "pdf", "docx", "jpg", "png", "json", "csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save file to upload directory
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Uploaded: {uploaded_file.name}")

# --- Chat Display ---
scroll_placeholder = st.empty()
chat_area = st.container()
with chat_area:
    st.markdown("<div class='scroll-container'>", unsafe_allow_html=True)

    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            content = msg["content"]
            # Detect category selection prompt
            if "please reply with the number (1-3)" in content.lower():
                # Show introductory text
                header_text = content.split("—")[0].strip()
                st.markdown(f"<div class='bot-msg'>{header_text}</div>", unsafe_allow_html=True)

                # Extract each numbered option line
                lines = content.split("\n")
                options = []
                for line in lines:
                    if line.strip().startswith(("1.", "2.", "3.")):
                        try:
                            number, text = line.split(". ", 1)
                            options.append((number.strip(), text.strip()))
                        except ValueError:
                            continue

                # Render options vertically
                for num, opt_text in options:
                    if st.button(opt_text, key=f"option_{num}_{len(st.session_state.history)}"):
                        # Add selected option as user message
                        st.session_state.history.append({"role": "user", "content": opt_text})
                        # Trigger re-run to send to backend as its number
                        st.session_state.pending_choice = num
                        st.rerun()
            else:
                # Normal bot messages
                st.markdown(f"<div class='bot-msg'>{content}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Input box ---
user_input = st.chat_input("Type your message here...")

# --- When user sends a message ---
if user_input:
    # Display user's message immediately
    st.session_state.history.append({"role": "user", "content": user_input})

    # Render chat again
    st.rerun()

# --- Process last user message ---
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    user_message = st.session_state.history[-1]["content"]
    session_id = st.session_state.session_id

    # Show typing animation (no Streamlit spinner)
    typing_placeholder = st.empty()
    for dots in ["", ".", "..", "..."]:
        typing_placeholder.markdown(f"<div class='typing'>🤖 Bot is typing{dots}</div>", unsafe_allow_html=True)
        time.sleep(0.3)

    # Send message to backend
    try:
        res = requests.post(
            f"{API_BASE}/chat/message",
            params={"session_id": session_id, "user_message": user_message},
            timeout=60
        )

        if res.status_code == 200:
            reply = res.json().get("reply", "Error: empty response")
        else:
            reply = f"⚠️ Error: {res.status_code} - {res.text}"
    except Exception as e:
        reply = f"⚠️ Could not connect to API: {e}"

    # Update chat with bot reply
    st.session_state.history.append({"role": "bot", "content": reply})

    # Remove typing animation once reply is received
    typing_placeholder.empty()

    # Refresh chat
    st.rerun()