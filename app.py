from config import S3_BUCKET_NAME
from src.generation import RAGPipeline
from src.embedding_creation import EmbeddingCreator, DocumentIndexManager
from src.document_processing import DocumentProcessor
import os
import sys
import time
import logging
import streamlit as st
import tempfile
from pathlib import Path
import streamlit.components.v1 as components

# Add the project directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page title and configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# JavaScript for copy to clipboard
COPY_BUTTON_JS = """
<script>
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(
        () => alert('Copied to clipboard!'),
        (err) => alert('Failed to copy: ' + err)
    );
}
</script>
"""

# Function to process an uploaded document
def process_document(uploaded_file, enforce_legal=True):
    """
    Process an uploaded document through the pipeline
    
    Args:
        uploaded_file (UploadedFile): Streamlit uploaded file
        enforce_legal (bool): If True, will reject non-legal documents
        
    Returns:
        dict: Processing result with status and metadata
    """
    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Progress bar for document processing
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate stages of processing
        status_text.text(f"Uploading {uploaded_file.name}...")
        progress_bar.progress(20)

        # Process document with validation
        document_processor = DocumentProcessor()
        document_id = Path(uploaded_file.name).stem

        status_text.text("Validating document...")
        progress_bar.progress(40)
        processed_file, is_legal, confidence, doc_type = document_processor.process_document(
            tmp_path,
            document_id=document_id,
            enforce_legal=enforce_legal
        )

        if not is_legal:
            st.warning(
                f"⚠️ Document '{uploaded_file.name}' doesn't appear to be a legal document (confidence: {confidence:.2f}). It has been processed, but results may not be optimal.")

        # Create embeddings
        status_text.text("Creating embeddings...")
        progress_bar.progress(60)
        embedding_creator = EmbeddingCreator()
        doc_id, embeddings_file, index_file = embedding_creator.create_embeddings(
            processed_file)

        # Update master index
        status_text.text("Updating index...")
        progress_bar.progress(80)
        index_manager = DocumentIndexManager()
        index_manager.create_or_update_master_index()

        # Complete progress
        status_text.text("Finalizing...")
        progress_bar.progress(100)

        # Add to session state
        doc_info = {
            'document_id': doc_id,
            'filename': uploaded_file.name,
            'processed_file': processed_file,
            'embeddings_file': embeddings_file,
            'index_file': index_file,
            'is_legal': is_legal,
            'legal_confidence': confidence,
            'document_type': doc_type
        }

        st.session_state.processed_docs.append(doc_info)

        # Initialize or reinitialize the RAG pipeline
        st.session_state.rag_pipeline = RAGPipeline()
        st.session_state.processing_complete = True

        # Clear progress bar and status
        progress_bar.empty()
        status_text.empty()

        return {
            'status': 'success',
            'document_id': doc_id,
            'is_legal': is_legal,
            'legal_confidence': confidence,
            'document_type': doc_type
        }
    except ValueError as e:
        st.error(f"❌ {str(e)}")
        return {
            'status': 'rejected',
            'error': str(e)
        }
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        logger.error(f"Error processing document: {str(e)}")
        logger.exception("Detailed traceback:")
        return {
            'status': 'error',
            'error': str(e)
        }
    finally:
        os.unlink(tmp_path)

# Function to delete a processed document
def delete_document(doc_id):
    """
    Remove a document from session state
    
    Args:
        doc_id (str): ID of the document to delete
    """
    st.session_state.processed_docs = [
        doc for doc in st.session_state.processed_docs if doc['document_id'] != doc_id
    ]
    # Reinitialize RAG pipeline if documents remain, else reset
    if st.session_state.processed_docs:
        st.session_state.rag_pipeline = RAGPipeline()
    else:
        st.session_state.rag_pipeline = None
        st.session_state.processing_complete = False
    st.rerun()

# Function to handle user queries
def process_query(query):
    """
    Process a user query through the RAG pipeline
    
    Args:
        query (str): User query
        
    Returns:
        dict: RAG pipeline response
    """
    if st.session_state.rag_pipeline is None:
        st.session_state.rag_pipeline = RAGPipeline()

    return st.session_state.rag_pipeline.process_query(query)

# Main app layout
st.title("📚 LexiSearch: Real-Time Legal Document Q&A")
st.markdown("""
Upload your legal PDFs and ask questions about their content. The system will use a 
Retrieval-Augmented Generation (RAG) approach to provide accurate answers based on the document content.
""")

# Sidebar for document upload and processing
with st.sidebar:
    st.header("Document Upload")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Select one or more files to upload"
    )

    # Real-time upload status
    if uploaded_files:
        upload_status = st.empty()
        for i, file in enumerate(uploaded_files):
            upload_status.text(f"Uploaded {file.name} ({i+1}/{len(uploaded_files)})")

    enforce_legal = st.checkbox("Enforce legal document validation", value=True)

    if uploaded_files:
        process_button = st.button("Process Documents")

        if process_button:
            successful_uploads = 0
            for uploaded_file in uploaded_files:
                result = process_document(
                    uploaded_file, enforce_legal=enforce_legal)
                if result['status'] == 'success':
                    successful_uploads += 1

            if successful_uploads > 0:
                st.success(
                    f"Successfully processed {successful_uploads} documents")
                upload_status.empty()

    st.divider()

    st.header("Processed Documents")
    if st.session_state.processed_docs:
        for doc in st.session_state.processed_docs:
            # Create color-coded badge with tooltip
            confidence = doc.get('legal_confidence', 1.0)
            if doc.get('is_legal', True):
                if confidence > 0.8:
                    confidence_badge = "🟢"
                    tooltip = "High confidence: Likely a legal document"
                elif confidence > 0.5:
                    confidence_badge = "🟡"
                    tooltip = "Moderate confidence: May be a legal document"
                else:
                    confidence_badge = "🟠"
                    tooltip = "Low confidence: Possibly not a legal document"
            else:
                confidence_badge = "🔴"
                tooltip = "Not recognized as a legal document"

            # Display document with badge and delete button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f'<span title="{tooltip}">{confidence_badge}</span> **{doc["filename"]}**',
                    unsafe_allow_html=True
                )
            with col2:
                if st.button("Delete", key=f"delete_{doc['document_id']}"):
                    delete_document(doc['document_id'])

            # Create expandable details section
            with st.expander("Details"):
                st.markdown(f"**Type:** {doc.get('document_type', 'Unknown')}")
                st.markdown(
                    f"**Legal Confidence:** {doc.get('legal_confidence', 0.0):.2f}")
                st.markdown(f"**Document ID:** {doc['document_id']}")
    else:
        st.info("No documents processed yet. Upload and process documents to begin.")

    st.divider()

    # Clear chat history and documents button
    if st.button("Clear Chat History and Documents"):
        st.session_state.chat_history = []
        st.session_state.processed_docs = []
        st.session_state.rag_pipeline = None
        st.session_state.processing_complete = False
        st.rerun()

# Main chat interface
st.header("Chat with your documents")

# Display chat history
components.html(COPY_BUTTON_JS)  # Include copy button JavaScript
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Add copy button for assistant responses
        if message["role"] == "assistant":
            if st.button("Copy Response", key=f"copy_{len(st.session_state.chat_history)}_{message['content'][:10]}"):
                st.markdown(
                    f'<button onclick="copyToClipboard(\'{message["content"].replace("'", "\\'")}\')">Copy</button>',
                    unsafe_allow_html=True
                )

        # If this is a response message with sources, show them in an expander
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(
                        f"**Source {i+1}** (Relevance: {source['score']:.2f})")
                    st.markdown(f"From document: *{source['document_id']}*")
                    st.markdown(f"```{source['text']}```")

# Query input - disabled if no documents are processed
query = st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.processed_docs
)

if query:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and display response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        typing_placeholder = st.empty()

        if not st.session_state.processed_docs:
            response = "Please upload and process documents before asking questions."
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})
            message_placeholder.markdown(response)
        else:
            try:
                with st.spinner("Generating response..."):
                    # Simulate typing animation
                    typing_placeholder.markdown("Typing...")
                    time.sleep(0.5)  # Brief delay for effect
                    typing_placeholder.empty()

                    # Process the query
                    result = process_query(query)

                    # Add response to chat history with sources
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['response'],
                        "sources": result['retrieved_chunks']
                    })

                    # Display response
                    message_placeholder.markdown(result['response'])

                    # Show retrieval info
                    with st.expander("View Sources"):
                        for i, chunk in enumerate(result['retrieved_chunks']):
                            st.markdown(
                                f"**Source {i+1}** (Relevance: {chunk['score']:.2f})")
                            st.markdown(
                                f"From document: *{chunk['document_id']}*")
                            st.markdown(f"```{chunk['text']}```")

                        st.info(
                            f"Retrieved in {result['retrieval_time']:.2f}s • Generated in {result['generation_time']:.2f}s • Total: {result['total_time']:.2f}s")

            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_message})
                message_placeholder.markdown(error_message)
                logger.error(error_message)
                typing_placeholder.empty()

# Footer
st.divider()
st.markdown("Built with Streamlit, LangChain, FAISS, and OpenAI")