import streamlit as st
import os
from datetime import datetime

# Placeholder functions for backend integration (replace with your actual RAG logic)
def process_pdfs(pdf_files):
    """
    Process uploaded PDFs and prepare them for RAG querying.
    Replace this with your actual text extraction and embedding logic.
    Returns a status message.
    """
    # Simulate processing
    for pdf_file in pdf_files:
        st.session_state['processed_files'].append(pdf_file.name)
    return "PDFs processed successfully. You can now ask questions."

def query_rag(user_query):
    """
    Query the RAG system with the user's question.
    Replace this with your actual RAG query function.
    Returns the response from the RAG system.
    """
    # Simulate a response (replace with your RAG output)
    return f"Response to '{user_query}': This is a placeholder answer from the RAG system."

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = []
if 'pdfs_uploaded' not in st.session_state:
    st.session_state['pdfs_uploaded'] = False

# Function to reset the session for a new chat
def start_new_chat():
    st.session_state['chat_history'] = []
    st.session_state['processed_files'] = []
    st.session_state['pdfs_uploaded'] = False
    st.success("Started a new chat. Please upload new PDFs to begin.")

# Streamlit app layout
st.set_page_config(page_title="LexiSearch: Legal Document Q&A", layout="wide")
st.title("LexiSearch: Real-Time Legal Document Q&A")
st.markdown("Upload your legal PDFs and ask questions about their content.")

# Sidebar for PDF upload, status, and new chat option
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze."
    )
    
    if uploaded_files and st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            status = process_pdfs(uploaded_files)
            st.session_state['pdfs_uploaded'] = True
            st.success(status)
    
    st.subheader("Processed Files")
    if st.session_state['processed_files']:
        for file_name in st.session_state['processed_files']:
            st.write(f"- {file_name}")
    else:
        st.write("No files processed yet.")
    
    st.subheader("Chat Options")
    if st.button("Start New Chat"):
        start_new_chat()

# Main chat interface
st.header("Chat with LexiSearch")
if not st.session_state['pdfs_uploaded']:
    st.warning("Please upload and process at least one PDF before asking questions.")
else:
    # Display chat history
    for entry in st.session_state['chat_history']:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
    
    # User input for new questions
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Add user question to chat history
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get RAG response
        with st.spinner("Generating response..."):
            response = query_rag(user_input)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
        
        # Update chat history
        st.session_state['chat_history'].append({
            "question": user_input,
            "answer": response,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

# Footer
st.markdown("---")
st.markdown("Developed by Hammad Sajid, Iqra Azfar, and Ali Muhammad Asad | CS 435 - Generative AI | Habib University")