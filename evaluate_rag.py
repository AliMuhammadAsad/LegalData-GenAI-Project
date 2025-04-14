from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from config import S3_BUCKET_NAME  # Assuming config is needed by imported modules
from src.generation import RAGPipeline
from src.embedding_creation import EmbeddingCreator, DocumentIndexManager
from src.document_processing import DocumentProcessor, process_documents
import os
import sys
import glob
import json
import csv
import time
import logging
import random
import re  # Moved import here
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- Add project root to path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# --------------------------------

# --- Project Imports ---
# -----------------------

# --- Hugging Face Imports ---
# --------------------------

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])  # Ensure logs go to stdout
logger = logging.getLogger(__name__)
# -------------------------

# --- Evaluation Configuration ---
EVALUATION_DOCS_DIR = os.path.join(current_dir, 'data', 'cases')
EVALUATION_OUTPUT_DIR = os.path.join(current_dir, 'evaluation_results')
QUESTIONS_PER_DOCUMENT = 3  # Reduced for faster testing, increase as needed
# --- IMPORTANT: Use Hugging Face Model Identifier ---
# Make sure you have access and resources for this model
LLM_JUDGE_MODEL_HF = "meta-llama/Meta-Llama-3-8B-Instruct"
LLM_QUESTION_GEN_MODEL_HF = "meta-llama/Meta-Llama-3-8B-Instruct"
# --- LLM Generation Config ---
# Max tokens for judge response (score + justification)
MAX_NEW_TOKENS_JUDGE = 200
MAX_NEW_TOKENS_QUESTIONS = 300  # Max tokens for question generation
LLM_TEMPERATURE = 0.1  # Low temp for deterministic generation/judging
# --- System Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
LOAD_IN_8BIT = True  # Set to True to load in 8-bit to save memory, requires bitsandbytes
# --- Retry Logic ---
MAX_RETRIES = 3  # Retries for LLM calls
RETRY_DELAY = 10  # Seconds between retries (increased for local model)
# ----------------------------

# --- Global Variables for Loaded Model ---
# To avoid reloading the model for every call
hf_pipeline = None
hf_model_name_loaded = None
# -----------------------------------------


def load_hf_model(model_name: str):
    """Loads the Hugging Face model and tokenizer."""
    global hf_pipeline, hf_model_name_loaded
    if hf_pipeline is not None and hf_model_name_loaded == model_name:
        logger.info(f"Model '{model_name}' already loaded.")
        return

    logger.info(
        f"Loading Hugging Face model '{model_name}' on device '{DEVICE}'...")
    logger.info(f"8-bit quantization: {LOAD_IN_8BIT}")

    quantization_config = None
    model_kwargs = {}
    if LOAD_IN_8BIT:
        if not torch.cuda.is_available():
            logger.warning(
                "BitsAndBytes 8-bit quantization requires CUDA. Loading in default precision.")
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = quantization_config
            # device_map="auto" is often needed for quantization with accelerate
            model_kwargs["device_map"] = "auto"

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        # If not using quantization or device_map, explicitly send to device later
        if "device_map" not in model_kwargs:
            model = AutoModelForCausalLM.from_pretrained(
                 model_name,
                 # bfloat16 for faster GPU inference if supported
                 torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                 **model_kwargs
             ).to(DEVICE)
        else:  # When using device_map="auto"
            model = AutoModelForCausalLM.from_pretrained(
                 model_name,
                 torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                 **model_kwargs
             )

        # Create pipeline
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # Use model's device if device_map was used
            device=model.device if hasattr(model, 'device') else DEVICE,
            # trust_remote_code=True # May be needed for some models
        )
        hf_model_name_loaded = model_name
        logger.info(
            f"Model '{model_name}' loaded successfully on device '{hf_pipeline.device}'.")

    except ImportError as e:
        logger.error(
            f"ImportError loading model: {e}. Make sure 'transformers', 'torch', 'accelerate', 'bitsandbytes' are installed.")
        raise
    except Exception as e:
        logger.error(
            f"Failed to load Hugging Face model '{model_name}': {e}", exc_info=True)
        raise


def format_llama3_instruct_prompt(prompt: str, system_message: str = None) -> str:
    """Formats the prompt according to Llama 3 Instruct template."""
    if system_message:
        # Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
        # Note: Tokenizer apply_chat_template is preferred if available and easy,
        # but manual formatting works too.
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
        formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        # Signal for assistant's turn
        formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_prompt


def call_llm(model_name: str, prompt: str, system_message: str = None, max_new_tokens: int = 200) -> str:
    """
    Calls the loaded Hugging Face LLM pipeline.
    """
    global hf_pipeline, hf_model_name_loaded
    if hf_pipeline is None or hf_model_name_loaded != model_name:
        # Attempt to load the required model if not already loaded
        try:
            load_hf_model(model_name)
        except Exception as e:
            raise Exception(
                f"Failed to load model '{model_name}' for LLM call: {e}") from e

    # Format prompt for Llama 3 Instruct
    formatted_prompt = format_llama3_instruct_prompt(prompt, system_message)

    logger.debug(
        f"Calling HF Pipeline {model_name}. Formatted Prompt Start: {formatted_prompt[:200]}...")

    for attempt in range(MAX_RETRIES):
        try:
            # --- Hugging Face Pipeline Generation ---
            outputs = hf_pipeline(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,  # Keep True even with low temp for slight variation if needed
                temperature=LLM_TEMPERATURE,
                top_p=0.9,  # Can adjust top_p as well
                eos_token_id=hf_pipeline.tokenizer.eos_token_id,
                pad_token_id=hf_pipeline.tokenizer.eos_token_id  # Often needed
            )
            # --- End HF Pipeline Generation ---

            if outputs and isinstance(outputs, list) and 'generated_text' in outputs[0]:
                # The pipeline output includes the prompt, so we need to extract only the generated part
                full_response = outputs[0]['generated_text']
                # Find the start of the assistant's response (after the final prompt marker)
                assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                marker_index = full_response.rfind(assistant_marker)
                if marker_index != -1:
                    generated_text = full_response[marker_index +
                        len(assistant_marker):].strip()
                    logger.debug(
                        f"LLM Raw Response: {generated_text[:100]}...")
                    return generated_text
                else:
                    # Fallback if marker isn't found (shouldn't happen with correct formatting)
                    logger.warning(
                        "Assistant marker not found in response, returning full output minus prompt.")
                    # This might be less reliable
                    # Estimate based on input
                    prompt_length = len(formatted_prompt)
                    return full_response[prompt_length:].strip()

            else:
                raise ValueError(
                    f"Unexpected LLM pipeline output format: {outputs}")

        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt + 1 == MAX_RETRIES:
                logger.error("LLM call failed after multiple retries.")
                raise Exception(f"LLM call failed: {e}") from e
            time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff

    # Should not be reached if retries fail, but added for safety
    raise Exception("LLM call failed after exhausting retries.")


def generate_questions_for_document(doc_content_sample: str, num_questions: int) -> List[str]:
    """
    Uses the HF LLM to generate relevant questions based on a sample of the document content.
    """
    logger.info(
        f"Generating {num_questions} questions using {LLM_QUESTION_GEN_MODEL_HF}...")
    sample_size = 4000  # Characters
    content_sample = doc_content_sample[:sample_size]

    system_message = "You are an assistant tasked with generating relevant questions about a legal document excerpt. Generate questions that test understanding of the key facts, definitions, obligations, or conclusions presented in the text. Focus ONLY on the provided text."
    prompt = f"""Based ONLY on the following legal document excerpt, generate {num_questions} distinct questions that someone might ask to understand its content. Output ONLY the questions, each on a new line, without numbering or any other text.

Document Excerpt:
---
{content_sample}
---

Questions:"""

    try:
        # Use the specific HF model identifier here
        response = call_llm(LLM_QUESTION_GEN_MODEL_HF, prompt,
                            system_message, max_new_tokens=MAX_NEW_TOKENS_QUESTIONS)
        questions = [q.strip() for q in response.split('\n') if q.strip() and q.strip(
        ) != "Questions:"]  # Filter out empty lines and potential headers
        if len(questions) < num_questions:
            logger.warning(
                f"LLM generated fewer questions ({len(questions)}) than requested ({num_questions}). Response: {response}")
        return questions[:num_questions]  # Return only the requested number
    except Exception as e:
        logger.error(f"Failed to generate questions: {e}")
        return []


def call_llama_judge(query: str, context: str, answer: str, aspect: str) -> Tuple[int, str]:
    """
    Uses the loaded HF Llama 3 8B Instruct to evaluate a specific aspect.
    """
    logger.info(f"Judging aspect '{aspect}' using {LLM_JUDGE_MODEL_HF}...")

    # System message and prompts remain the same as before...
    system_message = f"""You are an impartial evaluator assessing the quality of answers generated by a Retrieval-Augmented Generation (RAG) system for legal documents. Evaluate the answer based *only* on the provided context and the specific criteria for the aspect '{aspect}'. Provide a score from 1 to 5, where 1 is the worst and 5 is the best. Follow the scoring rubric strictly. After the score, provide a brief justification (1-2 sentences). Your output MUST start with "Score: [score]" followed by "Justification: [justification]". Do not include any other text before or after this format."""

    prompts = {
        "faithfulness": f"""
**Aspect: Faithfulness (Groundedness)**
**Scoring Rubric:**
1: The answer significantly contradicts the context or introduces substantial external information not found in the context.
2: The answer introduces some external information or makes claims not directly supported by the context.
3: The answer is mostly based on the context but might have minor unsupported details or slight inaccuracies.
4: The answer is accurately based on the context with no external information, but could be slightly more precise.
5: The answer is fully supported by the context, accurately reflects the information provided, and contains no external information.

**Provided Context:**
---
{context}
---
**Question:**
{query}
---
**Answer to Evaluate:**
{answer}
---
**Evaluation:**
Score: [1-5]
Justification: [Your brief justification]""",

        "answer_relevance": f"""
**Aspect: Answer Relevance**
**Scoring Rubric:**
1: The answer completely fails to address the question.
2: The answer addresses the question indirectly or tangentially, but is not a direct answer.
3: The answer addresses the question but includes significant irrelevant information.
4: The answer directly addresses the main point of the question but could be more focused.
5: The answer is directly and fully relevant to the question, addressing its core intent.

**Provided Context:**
---
{context}
---
**Question:**
{query}
---
**Answer to Evaluate:**
{answer}
---
**Evaluation:**
Score: [1-5]
Justification: [Your brief justification]""",

        "context_relevance": f"""
**Aspect: Context Relevance**
**Scoring Rubric:**
1: The retrieved context is completely irrelevant to the question.
2: The context contains some keywords but does not provide information needed to answer the question.
3: The context is partially relevant but misses key information or includes mostly irrelevant details.
4: The context is relevant and contains information needed to answer the question, but could be more focused or concise.
5: The context is highly relevant, concise, and contains all the necessary information to answer the question accurately.

**Provided Context:**
---
{context}
---
**Question:**
{query}
---
**Answer to Evaluate:** (Provided for reference, but evaluate the CONTEXT)
{answer}
---
**Evaluation:**
Score: [1-5]
Justification: [Your brief justification]""",

        "completeness": f"""
**Aspect: Completeness (Based ONLY on Context)**
**Scoring Rubric:**
1: The answer addresses only a minor part of the question, missing the main points found in the context.
2: The answer addresses some parts of the question but omits significant information available in the context.
3: The answer addresses the main parts of the question but lacks some details or nuances present in the context.
4: The answer addresses the question comprehensively based on the context but could be slightly more thorough.
5: The answer fully addresses all aspects of the question using the information available *within the provided context*. (Do not penalize if the context itself was incomplete).

**Provided Context:**
---
{context}
---
**Question:**
{query}
---
**Answer to Evaluate:**
{answer}
---
**Evaluation:**
Score: [1-5]
Justification: [Your brief justification]"""
    }

    if aspect not in prompts:
        raise ValueError(f"Unknown evaluation aspect: {aspect}")

    prompt = prompts[aspect]

    try:
        # Use the specific HF model identifier here
        response = call_llm(LLM_JUDGE_MODEL_HF, prompt,
                            system_message, max_new_tokens=MAX_NEW_TOKENS_JUDGE)
        return parse_judge_score(response, aspect)
    except Exception as e:
        logger.error(f"Failed to get judgment for aspect '{aspect}': {e}")
        return (0, f"Error during judgment: {e}")  # Return 0 score on error


def parse_judge_score(response: str, aspect: str) -> Tuple[int, str]:
    """Parses the 'Score: [score]\nJustification: [justification]' format."""
    try:
        # Make regex more robust to handle potential variations in spacing or newlines
        score_match = re.search(r"Score:\s*([1-5])", response, re.IGNORECASE)
        # Capture justification potentially spanning multiple lines, stopping at the end or next pattern
        justification_match = re.search(
            r"Justification:\s*(.*)", response, re.IGNORECASE | re.DOTALL)

        score = int(score_match.group(1)) if score_match else 0
        justification = justification_match.group(
            1).strip() if justification_match else "Parsing failed."

        if score == 0:
            logger.warning(
                f"Could not parse score for aspect '{aspect}' from response: {response}")
             # Provide more context in case of parsing failure
            justification = f"Score parsing failed. Raw response: '{response}'"
        elif not justification_match:
            logger.warning(
                f"Could not parse justification for aspect '{aspect}' from response: {response}")
            justification = f"Justification parsing failed. Raw response: '{response}'"

        return score, justification

    except Exception as e:
        logger.error(
            f"Error parsing judge response for aspect '{aspect}': {e}. Response: {response}")
        return (0, f"Error parsing response: {e}")

# --- Utility Functions (get_evaluation_documents, ensure_documents_processed, get_document_content, etc.) ---
# --- These functions remain the same as in the previous response. ---
# --- Make sure they are included here in the actual file. ---


def create_directory_if_not_exists(directory_path: str):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        logger.info(f"Creating directory: {directory_path}")
        os.makedirs(directory_path)


def get_evaluation_documents(docs_dir: str) -> List[str]:
    """Finds all PDF documents in the specified directory."""
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    docx_files = glob.glob(os.path.join(docs_dir, "*.docx"))
    txt_files = glob.glob(os.path.join(docs_dir, "*.txt"))
    all_files = pdf_files + docx_files + txt_files
    logger.info(
        f"Found {len(all_files)} documents (PDF, DOCX, TXT) for evaluation in {docs_dir}")
    return all_files


def ensure_documents_processed(doc_paths: List[str]) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Processes documents if not already processed and embedded.
    Returns paths to processed JSON files and validation results.
    """
    logger.info(
        "Ensuring all evaluation documents are processed and embedded...")
    processed_files_all = []
    validation_results_all = {}
    needs_processing = []
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    embeddings_dir = os.path.join(current_dir, 'data', 'embeddings')
    create_directory_if_not_exists(processed_dir)
    create_directory_if_not_exists(embeddings_dir)

    for doc_path in doc_paths:
        doc_id = Path(doc_path).stem
        processed_json_path = os.path.join(
            processed_dir, f"{doc_id}_processed.json")
        embeddings_json_path = os.path.join(
            embeddings_dir, f"{doc_id}_embeddings.json")
        # Assuming individual FAISS indices are not strictly necessary if master index is used
        # index_faiss_path = os.path.join(embeddings_dir, f"{doc_id}_index.faiss")

        # Check if processed file exists, assume embedding happened if processed exists
        if not os.path.exists(processed_json_path):
            logger.info(f"Document '{doc_id}' needs processing.")
            needs_processing.append(doc_path)
        else:
            logger.info(
                f"Document '{doc_id}' already processed. Skipping processing.")
            processed_files_all.append(processed_json_path)
             # Try to load validation results if file exists
            try:
                with open(processed_json_path, 'r', encoding='utf-8') as f:
                     data = json.load(f)
                     validation_results_all[doc_id] = data.get(
                         'validation_info', {'status': 'pre-existing'})
            except Exception as e:
                 logger.warning(
                     f"Could not load validation info for pre-existing {doc_id}: {e}")
                 validation_results_all[doc_id] = {
                     'status': 'pre-existing', 'error': 'Could not load info'}

    if needs_processing:
        logger.info(f"Processing {len(needs_processing)} documents...")
        # Use the updated process_documents which returns validation results
        # Assuming process_documents handles different file types based on extension
        processed_files_new, validation_results_new = process_documents(
            needs_processing, enforce_legal=False)  # Process all, even if flagged non-legal
        processed_files_all.extend(processed_files_new)
        validation_results_all.update(validation_results_new)

        if processed_files_new:
            logger.info("Creating embeddings for newly processed documents...")
            embedding_creator = EmbeddingCreator()
            embedding_results = []
            for proc_file in processed_files_new:
                try:
                     result = embedding_creator.create_embeddings(proc_file)
                     embedding_results.append(result)
                except Exception as e:
                     doc_id_err = Path(proc_file).stem.replace(
                         '_processed', '')
                     logger.error(
                         f"Failed to create embeddings for {doc_id_err}: {e}")
                     # Update validation result to show embedding error
                     if doc_id_err in validation_results_all:
                         validation_results_all[doc_id_err]['status'] = 'embedding_error'
                         validation_results_all[doc_id_err][
                             'error'] = f"Embedding failed: {e}"

            if embedding_results:
                logger.info("Updating master index...")
                index_manager = DocumentIndexManager()
                index_manager.create_or_update_master_index()
            else:
                logger.warning("No new embeddings were created, master index not updated.")
        else:
            logger.warning(
                "Document processing yielded no files for embedding.")

    # Ensure the master index exists even if no new docs were processed
    master_index_path = os.path.join(embeddings_dir, 'master_index.faiss')
    metadata_path = os.path.join(embeddings_dir, 'master_metadata.pkl')
    if not os.path.exists(master_index_path) or not os.path.exists(metadata_path):
        logger.info("Master index not found, creating/updating...")
        try:
             index_manager = DocumentIndexManager()
             index_manager.create_or_update_master_index()
        except Exception as e:
             logger.error(
                 f"Failed to create/update master index: {e}", exc_info=True)
             # Decide if evaluation should stop if index creation fails
             raise RuntimeError(
                 "Master index creation failed, cannot proceed with evaluation.") from e

    return processed_files_all, validation_results_all


def get_document_content(processed_json_path: str) -> str:
    """Loads the full text content from a processed JSON file."""
    try:
        with open(processed_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Concatenate text from all chunks
        full_text = "\n\n".join([chunk['text']
                                for chunk in data.get('chunks', [])])
        return full_text
    except FileNotFoundError:
        logger.error(f"Processed file not found: {processed_json_path}")
        return ""
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {processed_json_path}")
        return ""
    except Exception as e:
        logger.error(f"Failed to load content from {processed_json_path}: {e}")
        return ""


def evaluate_single_query(rag_pipeline: RAGPipeline, query: str, document_id: str) -> Dict[str, Any]:
    """Runs a query through RAG and then evaluates the result using LLM judge."""
    evaluation_result = {
        "document_id": document_id,
        "query": query,
        "answer": "",
        "context": "",
        "retrieval_time": 0.0,
        "generation_time": 0.0,
        "total_time": 0.0,
        "faithfulness_score": 0,
        "faithfulness_justification": "",
        "answer_relevance_score": 0,
        "answer_relevance_justification": "",
        "context_relevance_score": 0,
        "context_relevance_justification": "",
        "completeness_score": 0,
        "completeness_justification": "",
        "error": None
    }

    try:
        # 1. Run RAG Pipeline
        logger.info(
            f"Running RAG for query: '{query[:50]}...' on doc: {document_id}")
        rag_result = rag_pipeline.process_query(query)  # Assuming this returns the necessary dict
        evaluation_result.update({
            "answer": rag_result.get('response', 'N/A'),
            "retrieval_time": rag_result.get('retrieval_time', 0.0),
            "generation_time": rag_result.get('generation_time', 0.0),
            "total_time": rag_result.get('total_time', 0.0),
            # Format context for evaluation
            "context": rag_pipeline.retriever.format_context(rag_result.get('retrieved_chunks', []))
        })

        # Check if RAG pipeline itself returned an error state if applicable
        if rag_result.get('error'):
            raise Exception(f"RAG pipeline error: {rag_result['error']}")

        # Ensure context and answer are not empty before judging
        if not evaluation_result["context"]:
            logger.warning(f"Empty context retrieved for query: {query}. Skipping LLM judge.")
            evaluation_result["error"] = "Empty context retrieved"
             # Set scores to 0 or a specific code? Setting to 0.
            for aspect in ["faithfulness", "answer_relevance", "context_relevance", "completeness"]:
                 evaluation_result[f"{aspect}_score"] = 0
                 evaluation_result[f"{aspect}_justification"] = "Skipped due to empty context."
            return evaluation_result  # Return early

        if not evaluation_result["answer"] or evaluation_result["answer"] == 'N/A':
            logger.warning(f"Empty answer generated for query: {query}. Skipping LLM judge.")
            evaluation_result["error"] = "Empty answer generated"
            for aspect in ["faithfulness", "answer_relevance", "context_relevance", "completeness"]:
                 evaluation_result[f"{aspect}_score"] = 0
                 evaluation_result[f"{aspect}_justification"] = "Skipped due to empty answer."
            return evaluation_result  # Return early

        # 2. Evaluate Aspects using LLM Judge
        aspects = ["faithfulness", "answer_relevance",
            "context_relevance", "completeness"]
        for aspect in aspects:
            time.sleep(1)  # Small delay between judge calls
            score, justification = call_llama_judge(
                query=evaluation_result["query"],
                context=evaluation_result["context"],
                answer=evaluation_result["answer"],
                aspect=aspect
            )
            evaluation_result[f"{aspect}_score"] = score
            evaluation_result[f"{aspect}_justification"] = justification

    except Exception as e:
        logger.error(
            f"Error during evaluation for query '{query}': {e}", exc_info=True)
        evaluation_result["error"] = str(e)
        # Ensure default values remain if error occurs mid-evaluation
        evaluation_result["answer"] = evaluation_result.get(
            "answer", "ERROR DURING PROCESSING")

    return evaluation_result


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str):
    """Saves the evaluation results to a CSV file."""
    if not results:
        logger.warning("No results to save.")
        return

    # Ensure all keys are present in the header, even if some runs had errors
    fieldnames = [
        "document_id", "query", "answer", "context", "retrieval_time",
        "generation_time", "total_time", "faithfulness_score",
        "faithfulness_justification", "answer_relevance_score",
        "answer_relevance_justification", "context_relevance_score",
        "context_relevance_justification", "completeness_score",
        "completeness_justification", "error"
    ]
    # Add any potential missing keys from the first result (just in case)
    if results:
        for key in results[0].keys():
            if key not in fieldnames:
                fieldnames.append(key)

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')  # Ignore extra fields not in header
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Evaluation results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {e}")


def generate_summary_report(results: List[Dict[str, Any]], output_path: str):
    """Generates a Markdown summary report of the evaluation."""
    if not results:
        logger.warning("No results to generate report from.")
        return

    num_questions = len(results)
    avg_scores = {
        "faithfulness": 0,
        "answer_relevance": 0,
        "context_relevance": 0,
        "completeness": 0
    }
    valid_counts = {k: 0 for k in avg_scores}
    total_time = 0
    successful_evals = 0
    errors = 0

    for res in results:
        if res.get("error"):
            errors += 1
        else:
            successful_evals += 1
            total_time += res.get("total_time", 0)
            for aspect in avg_scores.keys():
                score = res.get(f"{aspect}_score", 0)
                if score > 0:  # Only average valid scores (parsed correctly, > 0)
                    avg_scores[aspect] += score
                    valid_counts[aspect] += 1

    report_content = f"# RAG System Evaluation Report\n\n"
    report_content += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_content += f"LLM Judge Model (Hugging Face): `{LLM_JUDGE_MODEL_HF}`\n"
    report_content += f"Evaluation Documents Path: `{EVALUATION_DOCS_DIR}`\n"
    report_content += f"Total Questions Attempted: {num_questions}\n"
    report_content += f"Successful RAG Pipeline Runs: {successful_evals}\n"
    report_content += f"Errors During Processing/Evaluation: {errors}\n\n"

    report_content += "## Overall Average Scores (1-5, higher is better)\n\n"
    if successful_evals > 0:
        avg_total_time = total_time / successful_evals
        report_content += f"- **Average Processing Time per Query (Successful Runs):** {avg_total_time:.2f} seconds\n"
    else:
        report_content += "- **Average Processing Time per Query:** N/A (No successful evaluations)\n"

    for aspect, total_score in avg_scores.items():
        count = valid_counts[aspect]
        if count > 0:
            avg = total_score / count
            report_content += f"- **{aspect.replace('_', ' ').title()}:** {avg:.2f} (based on {count} valid judge scores)\n"
        else:
            report_content += f"- **{aspect.replace('_', ' ').title()}:** N/A (No valid judge scores)\n"

    report_content += "\n*Note: Scores are averaged only over successful evaluations where the judge LLM provided a valid score (1-5). Errors during RAG processing or score parsing are excluded from averages.*\n"
    report_content += f"\nDetailed results can be found in `{os.path.basename(output_path.replace('.md', '.csv'))}`.\n"

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"Summary report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save summary report: {e}")

# --- Main Evaluation Function ---

def run_evaluation():
    """Orchestrates the entire evaluation process."""
    logger.info("--- Starting Large Scale RAG Evaluation ---")
    logger.info(f"Using Device: {DEVICE}")

    # 0. Load the LLM Model(s) first
    # Assuming judge and question gen use the same base model for simplicity now
    # If they differ, load both separately as needed.
    try:
        load_hf_model(LLM_JUDGE_MODEL_HF)  # Load the judge model (used for both here)
    except Exception as e:
        logger.error(
            f"Failed to load the primary LLM model: {e}. Evaluation cannot proceed.", exc_info=True)
        return

    # 1. Setup Output Directory
    create_directory_if_not_exists(EVALUATION_OUTPUT_DIR)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_output_path = os.path.join(
        EVALUATION_OUTPUT_DIR, f"evaluation_results_{timestamp}.csv")
    report_output_path = os.path.join(
        EVALUATION_OUTPUT_DIR, f"evaluation_summary_{timestamp}.md")

    # 2. Find and Process Documents
    doc_paths = get_evaluation_documents(EVALUATION_DOCS_DIR)
    if not doc_paths:
        logger.error("No documents found for evaluation. Exiting.")
        return

    try:
        processed_json_paths, validation_results = ensure_documents_processed(
            doc_paths)
    except RuntimeError as e:  # Catch index creation failure
        logger.error(f"Halting evaluation due to error during document processing/indexing: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error during document processing/indexing: {e}", exc_info=True)
        return  # Halt on other unexpected errors too

    if not processed_json_paths:
        logger.error("No documents could be processed or found pre-processed. Exiting.")
        return

    # 3. Initialize RAG Pipeline (after ensuring index exists)
    try:
        rag_pipeline = RAGPipeline()
    except Exception as e:
        logger.error(
            f"Failed to initialize RAG Pipeline: {e}. Check if master index exists.", exc_info=True)
        return

    # 4. Generate Questions and Evaluate
    all_evaluation_results = []
    for i, processed_path in enumerate(processed_json_paths):
        doc_id = Path(processed_path).stem.replace('_processed', '')
        logger.info(
            f"--- Evaluating Document {i+1}/{len(processed_json_paths)}: {doc_id} ---")

        doc_content = get_document_content(processed_path)
        if not doc_content:
            logger.warning(
                f"Skipping document {doc_id} due to content loading failure.")
            # Add a placeholder error result for this document
            all_evaluation_results.append({
                "document_id": doc_id, "query": "N/A", "answer": "N/A", "context": "N/A",
                "retrieval_time": 0, "generation_time": 0, "total_time": 0,
                "faithfulness_score": 0, "faithfulness_justification": "Content loading failed",
                "answer_relevance_score": 0, "answer_relevance_justification": "Content loading failed",
                "context_relevance_score": 0, "context_relevance_justification": "Content loading failed",
                "completeness_score": 0, "completeness_justification": "Content loading failed",
                "error": "Document content loading failed"
            })
            continue

        questions = generate_questions_for_document(
            doc_content, QUESTIONS_PER_DOCUMENT)
        if not questions:
            logger.warning(
                f"No questions generated for document {doc_id}. Skipping evaluation for this doc.")
            all_evaluation_results.append({
                "document_id": doc_id, "query": "N/A", "answer": "N/A", "context": "N/A",
                "retrieval_time": 0, "generation_time": 0, "total_time": 0,
                "faithfulness_score": 0, "faithfulness_justification": "Question generation failed",
                "answer_relevance_score": 0, "answer_relevance_justification": "Question generation failed",
                "context_relevance_score": 0, "context_relevance_justification": "Question generation failed",
                "completeness_score": 0, "completeness_justification": "Question generation failed",
                "error": "Question generation failed"
            })
            continue

        logger.info(f"Generated {len(questions)} questions for {doc_id}.")

        for q_idx, query in enumerate(questions):
            logger.info(
                f"--- Evaluating Q{q_idx+1}/{len(questions)} for Doc {doc_id} ---")
            eval_result = evaluate_single_query(rag_pipeline, query, doc_id)
            all_evaluation_results.append(eval_result)
            time.sleep(2)  # Add a small delay between full query evaluations

    # 5. Save Results
    save_results_to_csv(all_evaluation_results, csv_output_path)
    generate_summary_report(all_evaluation_results, report_output_path)

    logger.info("--- Evaluation Complete ---")


if __name__ == "__main__":
    # No extra imports needed here now as they are at the top
    run_evaluation()
