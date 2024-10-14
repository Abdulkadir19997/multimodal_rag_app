# Define the inference and preprocess API URLs

import streamlit as st
import requests
import base64
from io import BytesIO

# Define the inference and preprocess API URLs
PREPROCESS_API_URL = 'http://localhost:5004/app/demo/preprocess'
INFERENCE_API_URL = 'http://localhost:5004/app/demo/inference'

st.set_page_config(page_title="Multimodal RAG Chatbot App", layout="wide")

# Custom CSS for chat UI
st.markdown("""
    <style>
    .container {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    .chat-section {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        width: 100%; /* Extend to full width */
    }
    .chat-message {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
        display: inline-block;
    }
    .chat-user {
        text-align: left;
    }
    .chat-bot {
        text-align: right;
    }
    .bot-message {
        background-color: #e8f5e9;
    }
    </style>
""", unsafe_allow_html=True)

# Layout
st.title("RAG Chatbot with PDF Upload")
st.write("### Upload a PDF, ask questions, and view relevant documents!")

# Sidebar for PDF Upload
with st.sidebar:
    st.subheader("Upload PDF for Preprocessing")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Only preprocess when a PDF file is uploaded
    if uploaded_file:
        if st.button("Preprocess PDF"):
            with st.spinner('Processing the PDF...'):
                files = {'pdf_file': uploaded_file}
                response = requests.post(PREPROCESS_API_URL, files=files)
                if response.status_code == 200:
                    st.success(f"{uploaded_file.name} processed successfully")
                else:
                    st.error("An error occurred while processing the PDF.")

# Chat Section
st.subheader("Chat with the Bot")
chat_container = st.empty()  # Placeholder for chat messages

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Text input for the user query
user_query = st.text_input("Ask the bot a question:")
if st.button("Send"):
    if user_query:
        # Only trigger inference API here for chat interaction
        # Display user message
        st.session_state["chat_history"].append({"user": user_query})

        # Send user query to inference API
        with st.spinner('Getting the response...'):
            payload = {"query": user_query}
            response = requests.post(INFERENCE_API_URL, json=payload)

            if response.status_code == 200:
                response_json = response.json()
                # print(response_json)
                bot_response = response_json.get("response")
                docs_by_type = response_json.get("docs_by_type", {})

                # Append bot response to chat history
                st.session_state["chat_history"].append({"bot": bot_response})

                # Display similar results after the chat
                if "docs_by_type" in locals():
                    if "texts" in docs_by_type:
                        st.write("**Relevant Texts:**")
                        for text in docs_by_type["texts"]:
                            st.text_area("", text, height=300)

                    if "images" in docs_by_type:
                        st.write("**Relevant Images:**")
                        for img_base64 in docs_by_type["images"]:
                            img_data = base64.b64decode(img_base64)
                            st.image(img_data, use_column_width=True)
            else:
                st.error("An error occurred while fetching the response.")

# Display chat messages in a chat-like interface
with chat_container.container():
    for message in st.session_state["chat_history"]:
        if "user" in message:
            st.markdown(f"""
                <div class="chat-container chat-user">
                    <div class="chat-message">{message["user"]}</div>
                </div>
            """, unsafe_allow_html=True)
        elif "bot" in message:
            st.markdown(f"""
                <div class="chat-container chat-bot">
                    <div class="chat-message bot-message">{message["bot"]}</div>
                </div>
            """, unsafe_allow_html=True)

# Footer with extra instructions
st.write("---")
st.write("This is a demo app for Multimodal Retrieval-Augmented Generation (RAG) based chatbot.")