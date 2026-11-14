
import streamlit as st
import requests
import uuid


# -----------------------------
# Configuration
# -----------------------------

API_URL = "http://127.0.0.1:8000"


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="RecruitAI",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Session state
# -----------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("RecruitAI")

    st.caption("AI-powered recruiting assistant")

    st.divider()

    st.subheader("Document Management")

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "txt"]
    )

    strategy = st.selectbox(
        "Chunking strategy",
        ["recursive", "character"]
    )

    if st.button("Upload Document", use_container_width=True):

        if uploaded_file is None:
            st.warning("Please select a file first.")

        else:
            try:
                with st.spinner("Processing document..."):

                    response = requests.post(
                        f"{API_URL}/api/v1/documents/ingest",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                            )
                        },
                        data={
                            "strategy": strategy
                        },
                        timeout=120
                    )

                    response.raise_for_status()

                    result = response.json()

                st.success("Document uploaded successfully!")

                st.write("Filename:", result["filename"])
                st.write("Chunks stored:", result["chunks_stored"])

            except requests.RequestException as e:
                st.error(f"Upload failed: {e}")

    st.divider()

    st.subheader("Conversation")

    st.caption("Session ID:")
    st.code(st.session_state.session_id)

    if st.button("New Chat", use_container_width=True):

        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []

        st.rerun()


# -----------------------------
# Main chat interface
# -----------------------------

st.title("RecruitAI Assistant")

st.caption(
    "Ask questions about your uploaded documents "
    "or book an interview using natural language."
)


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat input
# -----------------------------

query = st.chat_input("Ask me anything...")


if query:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Send query to FastAPI
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = requests.post(
                    f"{API_URL}/api/v1/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "query": query
                    },
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()

                answer = result["answer"]

                st.markdown(answer)

                # Display booking information if available
                booking = result.get("booking")

                if booking:

                    st.success("Interview booked successfully!")

                    st.write("Booking ID:", booking["id"])
                    st.write("Name:", booking["name"])
                    st.write("Email:", booking["email"])
                    st.write("Date:", booking["date"])
                    st.write("Time:", booking["time"])

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except requests.RequestException as e:

                st.error(f"Could not connect to backend: {e}")
