import streamlit as st
import tempfile
from typing import List

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="wide"
)

def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded files and return their paths"""
    temp_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_paths.append(tmp_file.name)
    return temp_paths

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Page Header
st.title("📚 Document Q&A System")
st.write("Upload PDFs and ask questions about their content")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )
    
    if uploaded_files:
        new_files = [f.name for f in uploaded_files if f.name not in [f.name for f in st.session_state.uploaded_files]]
        if new_files:
            st.session_state.uploaded_files.extend(uploaded_files)
            temp_paths = save_uploaded_files(uploaded_files)
            st.success(f"Successfully uploaded {len(new_files)} new files!")
    
    # Display currently loaded documents
    if st.session_state.uploaded_files:
        st.write("📂 Loaded Documents:")
        for file in st.session_state.uploaded_files:
            st.write(f"- {file.name}")

# Main chat interface
st.divider()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if not st.session_state.uploaded_files:
    st.info("👆 Please upload PDF documents to start asking questions")
else:
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Here you would integrate your RAG pipeline
                # response = your_rag_function(prompt, pdf_paths)
                response = "This is a placeholder response. Replace this with your actual RAG implementation."
                st.write(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# Clear chat button
if st.session_state.chat_history:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()