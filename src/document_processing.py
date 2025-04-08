from src.utils import get_aws_client, upload_file_to_s3, create_directory_if_not_exists
from config import S3_BUCKET_NAME, CHUNK_SIZE, CHUNK_OVERLAP
import os
import time
import json
import logging
import PyPDF2
import re
from io import BytesIO
import sys
from src.legal_utils import legal_text_chunker, extract_legal_metadata, identify_document_type
from config import LEGAL_CHUNK_SIZE, LEGAL_CHUNK_OVERLAP
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.s3_client = get_aws_client('s3')
        self.textract_client = get_aws_client('textract')

    def process_document(self, file_path, document_id=None, enforce_legal=True):
        """
        Process a document through the complete pipeline:
        1. Upload to S3
        2. Extract text using Textract
        3. Validate if it's a legal document
        4. Chunk the text
        5. Save processed text
        
        Args:
            file_path (str): Path to the document file
            document_id (str, optional): Unique identifier for the document.
                                         If not provided, file name is used.
            enforce_legal (bool): If True, will raise an error for non-legal documents
        
        Returns:
            tuple: (processed_text_path, is_legal, legal_confidence, document_type)
        """
        if document_id is None:
            document_id = os.path.splitext(os.path.basename(file_path))[0]

        # Upload document to S3
        s3_key = f"documents/{document_id}/{os.path.basename(file_path)}"
        if not upload_file_to_s3(file_path, s3_key):
            raise Exception(f"Failed to upload document {file_path} to S3")

        # Extract text from document
        extracted_text = self._extract_text_from_document(s3_key)

        # Validate if it's a legal document
        is_legal, legal_confidence, document_type = self.validate_legal_document(
            extracted_text)

        logger.info(
            f"Document validation: is_legal={is_legal}, confidence={legal_confidence:.2f}, type={document_type}")

        if enforce_legal and not is_legal:
            raise ValueError(
                f"The document does not appear to be a legal document (confidence: {legal_confidence:.2f}). Processing aborted.")

        # Chunk the extracted text
        text_chunks = self._chunk_text(extracted_text)

        # Save processed text with validation information
        processed_text_path = self._save_processed_text(text_chunks, document_id, {
            'is_legal': is_legal,
            'legal_confidence': legal_confidence,
            'document_type': document_type
        })

        return (processed_text_path, is_legal, legal_confidence, document_type)

    def _extract_text_from_document(self, s3_key):
        """
        Extract text from a document using AWS Textract
        
        Args:
            s3_key (str): S3 object key of the document
            
        Returns:
            str: Extracted text
        """
        logger.info(f"Extracting text from document {s3_key}")

        # Check if it's a PDF (we can do simple PDF extraction for text PDFs)
        if s3_key.lower().endswith('.pdf'):
            try:
                # Try direct PDF extraction first (faster for text PDFs)
                return self._extract_text_from_pdf(s3_key)
            except Exception as e:
                logger.warning(
                    f"Direct PDF extraction failed: {str(e)}. Falling back to Textract.")

        # Start Textract job
        response = self.textract_client.start_document_text_detection(
            DocumentLocation={'S3Object': {
                'Bucket': S3_BUCKET_NAME, 'Name': s3_key}}
        )
        job_id = response['JobId']
        logger.info(f"Started Textract job {job_id} for document {s3_key}")

        # Poll for job completion
        status = 'IN_PROGRESS'
        while status == 'IN_PROGRESS':
            time.sleep(5)
            response = self.textract_client.get_document_text_detection(
                JobId=job_id)
            status = response['JobStatus']
            logger.info(f"Textract job status: {status}")

        if status == 'FAILED':
            raise Exception(
                f"Textract job failed: {response.get('StatusMessage', 'Unknown error')}")

        # Get results
        pages_text = []
        response = self.textract_client.get_document_text_detection(
            JobId=job_id)
        pages_text.extend(
            [item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE'])

        # Handle pagination for large documents
        while 'NextToken' in response:
            response = self.textract_client.get_document_text_detection(
                JobId=job_id, NextToken=response['NextToken']
            )
            pages_text.extend(
                [item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE'])

        return '\n'.join(pages_text)

    def _extract_text_from_pdf(self, s3_key):
        """
        Extract text directly from a PDF document using PyPDF2
        
        Args:
            s3_key (str): S3 object key of the PDF document
            
        Returns:
            str: Extracted text
        """
        # Download file from S3 to memory
        response = self.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        pdf_content = BytesIO(response['Body'].read())

        # Extract text from PDF
        text = []
        # Correct way to use PdfReader
        pdf = PyPDF2.PdfReader(pdf_content)
        for page_num in range(len(pdf.pages)):
            text.append(pdf.pages[page_num].extract_text())

        return '\n'.join(text)

    def _chunk_text(self, text, chunk_size=LEGAL_CHUNK_SIZE, chunk_overlap=LEGAL_CHUNK_OVERLAP):
        """
        Split text into overlapping chunks optimized for legal documents
        
        Args:
            text (str): Text to split
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            list: List of text chunks
        """
        if not text:
            return []

        # Extract legal metadata
        legal_metadata = extract_legal_metadata(text)

        # Use legal-specific chunking
        chunks = legal_text_chunker(
            text, max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Add metadata to first chunk if we have chunks
        if chunks and legal_metadata:
            metadata_str = "Document Metadata:\n"
            metadata_str += f"Document Type: {legal_metadata['document_type']}\n"

            if legal_metadata['parties']:
                metadata_str += f"Parties: {', '.join(legal_metadata['parties'])}\n"

            if legal_metadata['dates']:
                metadata_str += f"Key Dates: {', '.join(legal_metadata['dates'])}\n"

            chunks[0] = metadata_str + "\n" + chunks[0]

        return chunks

    def _save_processed_text(self, text_chunks, document_id, validation_info=None):
        """
        Save processed text chunks to a file
        
        Args:
            text_chunks (list): List of text chunks
            document_id (str): Document identifier
            validation_info (dict, optional): Legal document validation info
            
        Returns:
            str: Path to the saved processed text file
        """
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'data', 'processed')
        create_directory_if_not_exists(processed_dir)

        output_file = os.path.join(
            processed_dir, f"{document_id}_processed.json")

        # Save chunks with metadata
        chunks_with_metadata = []
        for i, chunk in enumerate(text_chunks):
            chunks_with_metadata.append({
                "chunk_id": f"{document_id}_chunk_{i}",
                "document_id": document_id,
                "chunk_index": i,
                "text": chunk
            })

        # Add validation info to the output
        output_data = {
            "document_id": document_id,
            "chunks": chunks_with_metadata,
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_info": validation_info or {}
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(
            f"Saved {len(text_chunks)} processed text chunks to {output_file}")

        # Upload processed text to S3
        s3_key = f"processed/{document_id}_processed.json"
        upload_file_to_s3(output_file, s3_key)

        return output_file

    # Add this as a method to the DocumentProcessor class


    def validate_legal_document(self, text):
        """
            Validate if a document appears to be a legal document

            Args:
                text (str): Document text

            Returns:
                tuple: (is_legal, confidence, document_type)
            """
        # Check for minimum legal indicators
        legal_indicators = [
            r'\b(?:contract|agreement|terms|conditions|clause|provision|party|parties)\b',
            r'\b(?:law|legal|court|judicial|statute|regulation|pursuant|hereby)\b',
                r'\b(?:witness|whereof|executed|signed|notarized|certified)\b',
                r'(?:section|article)\s+\d+',
                r'\d+\.\d+\s+[A-Z][a-z]+'  # Numbered clauses
            ]

        # Count how many indicators are found
        indicator_count = 0
        for pattern in legal_indicators:
                if re.search(pattern, text, re.IGNORECASE):
                    indicator_count += 1

            # Get document type
        doc_type = identify_document_type(text)

            # Calculate confidence (0-1)
        confidence = indicator_count / len(legal_indicators)

            # Document is considered legal if:
            # 1. It has a specific legal document type OR
            # 2. It has at least 3 legal indicators
        is_legal = (doc_type != "Unknown Legal Document") or (
             indicator_count >= 3)

        return (is_legal, confidence, doc_type)

# Function to process multiple documents

# Replace the process_documents function with this updated version


def process_documents(file_paths, enforce_legal=True):
    """
    Process multiple documents
    
    Args:
        file_paths (list): List of paths to document files
        enforce_legal (bool): If True, will skip non-legal documents
        
    Returns:
        tuple: (processed_files, validation_results)
            - processed_files: List of paths to processed text files
            - validation_results: Dict mapping document IDs to validation results
    """
    processor = DocumentProcessor()
    processed_files = []
    validation_results = {}

    for file_path in file_paths:
        try:
            document_id = os.path.splitext(os.path.basename(file_path))[0]

            # Process document with validation
            processed_file, is_legal, confidence, doc_type = processor.process_document(
                file_path,
                document_id=document_id,
                enforce_legal=enforce_legal
            )

            processed_files.append(processed_file)
            validation_results[document_id] = {
                'is_legal': is_legal,
                'confidence': confidence,
                'document_type': doc_type,
                'status': 'processed' if is_legal else 'processed_with_warning'
            }

            logger.info(
                f"Document {file_path} processed: {validation_results[document_id]}")

        except ValueError as e:
            # This is raised when enforce_legal=True and document is not legal
            logger.warning(f"Document {file_path} skipped: {str(e)}")
            document_id = os.path.splitext(os.path.basename(file_path))[0]
            validation_results[document_id] = {
                'is_legal': False,
                'status': 'skipped',
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            document_id = os.path.splitext(os.path.basename(file_path))[0]
            validation_results[document_id] = {
                'status': 'error',
                'error': str(e)
            }

    return processed_files, validation_results
