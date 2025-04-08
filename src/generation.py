from config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS
import os
import logging
import time
from openai import OpenAI
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResponseGenerator:
    def __init__(self, api_key=OPENAI_API_KEY, model_name=MODEL_NAME):
        """
        Initialize the response generator
        
        Args:
            api_key (str): OpenAI API key
            model_name (str): Name of the OpenAI model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Response generator initialized with model: {model_name}")

    def generate_response(self, query, context, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
        """
        Generate a response to the query based on the provided context
        
        Args:
            query (str): User query
            context (str): Context from retrieved documents
            temperature (float): Controls randomness of the output
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Generated response
        """
        start_time = time.time()

        # Create the prompt
        prompt = self._create_prompt(query, context)

        # Call the OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a specialized legal document assistant that helps users understand the content of legal documents. You provide factual information based strictly on the provided context and explain legal concepts in clear terms. You do not provide legal advice or opinions. If the context doesn't contain the information needed to answer the question, you acknowledge that and don't make up information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract the response text
            response_text = response.choices[0].message.content

            # Log the response time
            elapsed_time = time.time() - start_time
            logger.info(f"Response generated in {elapsed_time:.2f} seconds")

            return response_text

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Sorry, I was unable to generate a response. Error: {str(e)}"

    def _create_prompt(self, query, context):
        """
        Create a prompt for the LLM based on the query and context,
        optimized for legal document understanding
        
        Args:
            query (str): User query
            context (str): Context from retrieved documents
            
        Returns:
            str: Formatted prompt
        """
        return f"""
    You are a legal information assistant that helps users understand legal documents. Please answer the following question based on the provided context information from legal documents.

    Question: {query}

    Context Information from Legal Documents:
    {context}

    Instructions for responding:
    1. Answer the question strictly based on the provided context.
    2. If the context doesn't contain enough information to provide a complete answer, acknowledge the limitations of the information provided.
    3. Use legal terminology appropriately when it appears in the context.
    4. Explain legal concepts mentioned in the context in clear, understandable language.
    5. When referencing specific parts of legal documents (sections, clauses, etc.), cite them precisely if the information is available in the context.
    6. Avoid providing legal advice or opinions - focus on explaining what the documents contain.
    7. If the question asks for legal advice rather than document content, clarify that you can only provide information about the documents, not legal advice.

    Remember to keep your response focused on the document content and avoid speculating beyond what's provided in the context.
    """

# Create a complete RAG pipeline


class RAGPipeline:
    def __init__(self):
        """
        Initialize the RAG pipeline
        """
        from src.retrieval import DocumentRetriever

        self.retriever = DocumentRetriever()
        self.generator = ResponseGenerator()
        logger.info("RAG pipeline initialized")

    def process_query(self, query, top_k=5):
        """
        Process a user query through the complete RAG pipeline
        
        Args:
            query (str): User query
            top_k (int): Number of top results to retrieve
            
        Returns:
            dict: Dictionary containing the response and metadata
        """
        logger.info(f"Processing query: {query}")

        # Retrieve relevant document chunks
        start_time = time.time()
        results = self.retriever.search(query, top_k=top_k)
        retrieval_time = time.time() - start_time

        # Format the context
        context = self.retriever.format_context(results)

        # Generate response
        start_time = time.time()
        response = self.generator.generate_response(query, context)
        generation_time = time.time() - start_time

        # Return response with metadata
        return {
            'query': query,
            'response': response,
            'retrieved_chunks': results,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + generation_time
        }
