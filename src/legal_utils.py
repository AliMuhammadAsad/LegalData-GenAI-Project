import re
import logging
import nltk
from nltk.tokenize import sent_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Common legal document sections
LEGAL_SECTIONS = [
    "PREAMBLE", "WITNESSETH", "RECITALS", "DEFINITIONS", "INTERPRETATION",
    "REPRESENTATIONS AND WARRANTIES", "COVENANTS", "CONDITIONS PRECEDENT",
    "INDEMNIFICATION", "TERM AND TERMINATION", "GOVERNING LAW",
    "DISPUTE RESOLUTION", "MISCELLANEOUS", "EXHIBITS", "SCHEDULES",
    "APPENDIX", "ANNEX"
]

# Common legal document types
LEGAL_DOCUMENT_TYPES = [
    "CONTRACT", "AGREEMENT", "LEASE", "LICENSE", "AMENDMENT",
    "DEED", "WILL", "TRUST", "POWER OF ATTORNEY", "MEMORANDUM",
    "BYLAW", "STATUTE", "REGULATION", "CODE", "ACT",
    "ORDINANCE", "CONSTITUTION", "TREATY", "BRIEF", "OPINION",
    "MOTION", "PETITION", "COMPLAINT", "ANSWER", "JUDGMENT",
    "ORDER", "DECREE", "INJUNCTION", "SUBPOENA", "WARRANT"
]

# Legal citation patterns
LEGAL_CITATION_PATTERNS = [
    r'\d+\s+U\.S\.\s+\d+',  # US Reports
    r'\d+\s+S\.\s*Ct\.\s+\d+',  # Supreme Court Reporter
    r'\d+\s+F\.\s*(?:Supp\.|App\'x|)\s*\d+',  # Federal Reporter
    r'\d+\s+F\.\s*\d+d\s+\d+',  # Federal Reporter (2d, 3d)
    r'[A-Za-z]+\s+v\.\s+[A-Za-z]+',  # Case names
]


def identify_document_type(text):
    """
    Attempt to identify the type of legal document
    
    Args:
        text (str): Document text
        
    Returns:
        str: Identified document type or "Unknown Legal Document"
    """
    # Check first 1000 characters for document type
    header = text[:1000].upper()

    for doc_type in LEGAL_DOCUMENT_TYPES:
        if doc_type in header or doc_type.replace(" ", "") in header.replace(" ", ""):
            return doc_type

    # Check for common patterns
    if "PARTIES:" in header or "BETWEEN:" in header:
        return "CONTRACT/AGREEMENT"

    if "CASE NO" in header or "DOCKET NO" in header:
        return "LEGAL FILING"

    if "OPINION" in header or "DISSENT" in header:
        return "COURT OPINION"

    if "STATUTE" in header or "PUBLIC LAW" in header:
        return "LEGISLATION"

    return "Unknown Legal Document"


def extract_legal_metadata(text):
    """
    Extract metadata from a legal document
    
    Args:
        text (str): Document text
        
    Returns:
        dict: Document metadata
    """
    metadata = {
        "document_type": identify_document_type(text),
        "parties": [],
        "dates": [],
        "citations": []
    }

    # Extract parties (simple approach)
    party_matches = re.findall(
        r'(?:BETWEEN|PARTIES):\s*([^,]*?)\s+AND\s+([^,\n]*)', text[:2000])
    if party_matches:
        metadata["parties"] = list(party_matches[0])

    # Extract dates
    date_matches = re.findall(
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', text)
    metadata["dates"] = date_matches[:5]  # Limit to first 5 dates

    # Extract citations
    for pattern in LEGAL_CITATION_PATTERNS:
        citation_matches = re.findall(pattern, text)
        # Limit to first 10 citations
        metadata["citations"].extend(citation_matches[:10])

    return metadata


def legal_text_chunker(text, max_chunk_size=1000, chunk_overlap=200):
    """
    Chunk legal text considering document structure
    
    Args:
        text (str): Document text
        max_chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    # Try to break at meaningful section boundaries
    section_pattern = r'(?:\n\s*|^)(?:' + \
        '|'.join(LEGAL_SECTIONS) + r')(?:\s*\n|\s*$)'
    sections = re.split(section_pattern, text, flags=re.IGNORECASE)

    chunks = []
    for section in sections:
        if not section.strip():
            continue

        if len(section) <= max_chunk_size:
            chunks.append(section.strip())
        else:
            # Further break section into paragraphs
            paragraphs = re.split(r'\n\s*\n', section)

            current_chunk = []
            current_size = 0

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                paragraph_size = len(paragraph)

                if current_size + paragraph_size <= max_chunk_size:
                    current_chunk.append(paragraph)
                    current_size += paragraph_size
                elif paragraph_size > max_chunk_size:
                    # If paragraph is too large, break it into sentences
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_size = 0

                    sentences = sent_tokenize(paragraph)
                    sentence_chunk = []
                    sentence_chunk_size = 0

                    for sentence in sentences:
                        sentence = sentence.strip()
                        sentence_size = len(sentence)

                        if sentence_chunk_size + sentence_size <= max_chunk_size:
                            sentence_chunk.append(sentence)
                            sentence_chunk_size += sentence_size
                        else:
                            if sentence_chunk:
                                chunks.append(" ".join(sentence_chunk))
                            sentence_chunk = [sentence]
                            sentence_chunk_size = sentence_size

                    if sentence_chunk:
                        chunks.append(" ".join(sentence_chunk))
                else:
                    # Add current chunk and start a new one
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [paragraph]
                    current_size = paragraph_size

            # Don't forget the last chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

    # Add document type info to the first chunk
    if chunks:
        doc_type = identify_document_type(text)
        chunks[0] = f"[Document Type: {doc_type}]\n\n" + chunks[0]

    return chunks
