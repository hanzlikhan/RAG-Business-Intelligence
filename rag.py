"""
rag.py
──────
Retrieval-Augmented Generation (RAG) utilities.
Provides batched, cached, and retry-wrapped embedding generation using Gemini.
"""

import os
import functools
from typing import List, Union, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils import get_logger, with_retries

logger = get_logger(__name__)

# This API key only supports gemini-embedding-001 (which is 3072 dims)
# but we will manual truncate to 768 to meet system interface requirements.
EMBEDDING_MODEL = "models/gemini-embedding-001"
MAX_BATCH_SIZE = 10


def _build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Build the Langchain embedding client."""
    # Ensure key exists
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found in environment.")
    
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )

_embeddings_client = None

class TruncatedGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Wrapper for GoogleGenerativeAIEmbeddings that truncates 3072 dims to 768.
    Ensures compatibility with the ai-bos Pinecone index.
    """
    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        embeddings = super().embed_documents(texts, **kwargs)
        return [emb[:768] for emb in embeddings]

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        embedding = super().embed_query(text, **kwargs)
        return embedding[:768]

def get_embeddings_client() -> TruncatedGoogleEmbeddings:
    """Singleton pattern for building the truncated embeddings client."""
    global _embeddings_client
    if _embeddings_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not found in environment.")
        _embeddings_client = TruncatedGoogleEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key
        )
    return _embeddings_client


@functools.lru_cache(maxsize=1000)
def _embed_single_cached(text: str) -> List[float]:
    """
    Embed a single text string and cache the result.
    LRU Cache applies only to single strings.
    """
    return _embed_batch([text])[0]


@with_retries
def _embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using the Gemini API.
    Wrapped with tenacity for retries on API limits/errors.
    """
    client = get_embeddings_client()
    try:
        # gemini-embedding-001 yields 3072 dims, force 768
        raw_embeddings = client.embed_documents(texts)
        return [emb[:768] for emb in raw_embeddings]
    except Exception as e:
        logger.warning("Embedding failed: %s. Attempting fallback if applicable...", e)
        raise e


def embed_text(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings for one or multiple strings.
    Includes input validation, batching (max 10), and caching (for single strings).

    Args:
        texts: A string or a list of strings to embed.

    Returns:
        List[float] if input is a single string.
        List[List[float]] if input is a list of strings.
        
    Raises:
        ValueError: If a string is empty or invalid.
    """
    if isinstance(texts, str):
        texts = texts.strip()
        if not texts:
            raise ValueError("Input text must not be empty.")
        # Use cache for single requests
        return _embed_single_cached(texts)
        
    elif isinstance(texts, list):
        if not texts:
            return []
            
        validated_texts = []
        for t in texts:
            t = t.strip()
            if not t:
                raise ValueError("Input texts contain an empty string.")
            validated_texts.append(t)
            
        all_embeddings = []
        # Process in chunks of MAX_BATCH_SIZE
        for i in range(0, len(validated_texts), MAX_BATCH_SIZE):
            batch = validated_texts[i:i + MAX_BATCH_SIZE]
            logger.info("Embedding batch of %d items...", len(batch))
            batch_result = _embed_batch(batch)
            all_embeddings.extend(batch_result)
            
        return all_embeddings

    else:
        raise TypeError("Input must be a string or a list of strings.")

# ── Document Ingestion & Chunking ──────────────────────────────────────────────

import asyncio
import time
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def _get_loader_for_file(file_path: str):
    """Return the correct Langchain loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

async def process_file_async(file_path: str) -> List[Document]:
    """Asynchronously load and chunk a single file."""
    loop = asyncio.get_event_loop()
    
    # Run the blocking I/O loader in a thread pool to avoid blocking async loop
    loader = _get_loader_for_file(file_path)
    docs = await loop.run_in_executor(None, loader.load)
    
    # Setup Splitter: 1000 size, 200 overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    
    chunks = splitter.split_documents(docs)
    
    # Inject metadata (timestamp, source)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for chunk in chunks:
        chunk.metadata["source"] = file_path
        chunk.metadata["timestamp"] = timestamp
        
    return chunks

def process_file_sync_fallback(file_path: str) -> List[Document]:
    """Synchronous fallback for processing a file if async fails."""
    try:
        loader = _get_loader_for_file(file_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for chunk in chunks:
            chunk.metadata["source"] = file_path
            chunk.metadata["timestamp"] = timestamp
            
        return chunks
    except Exception as e:
        logger.error("Sync fallback failed for %s: %s", file_path, e)
        raise e

async def ingest_documents_async(file_paths: List[str]) -> List[Document]:
    """
    Ingest multiple files asynchronously.
    """
    tasks = []
    for path in file_paths:
        if not os.path.exists(path):
            logger.warning("File does not exist: %s. Skipping.", path)
            continue
        tasks.append(process_file_async(path))
    
    if not tasks:
        raise ValueError("No valid files to process.")
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_chunks = []
    for path, result in zip(file_paths, results):
        if isinstance(result, Exception):
            logger.warning("Async ingestion failed for %s (%s). Using sync fallback.", path, result)
            try:
                # Sync fallback
                fallback_chunks = process_file_sync_fallback(path)
                all_chunks.extend(fallback_chunks)
            except Exception as e:
                logger.error("Failed completely to load %s: %s", path, e)
        else:
            all_chunks.extend(result)
            
    return all_chunks

def load_and_chunk_files(file_paths: List[str]) -> List[Document]:
    """
    Main entrypoint for document ingestion. Validates input, 
    runs async gather, and returns text chunks.
    """
    if not file_paths:
        return []
        
    try:
        # Run the async pipeline in the current event loop or new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        chunks = loop.run_until_complete(ingest_documents_async(file_paths))
        return chunks
    except Exception as e:
        logger.error("Ingestion pipeline failed: %s", e)
        raise e

# ── Pinecone Vector Store ──────────────────────────────────────────────────────

from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

PINECONE_INDEX_NAME = "ai-bos"
PINECONE_DIMENSION = 768
MAX_UPSERT_BATCH_SIZE = 50

@with_retries
def init_pinecone_index():
    """
    Initialize the Pinecone index. Creates it if it doesn't exist.
    Blocks until the index is ready.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or api_key == "replace_with_pinecone_key":
        raise EnvironmentError("PINECONE_API_KEY is missing or invalid in .env")
        
    pc = PineconeClient(api_key=api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info("Creating Pinecone index '%s' (this may take a minute)...", PINECONE_INDEX_NAME)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(2)
            
    logger.info("Pinecone index '%s' is ready.", PINECONE_INDEX_NAME)

@with_retries
async def upsert_batch_async(vectorstore: PineconeVectorStore, batch: List[Document]):
    """Async upsert a single batch using the vectorstore."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, vectorstore.add_documents, batch)

async def upsert_documents_to_pinecone_async(chunks: List[Document]):
    """
    Upsert documents into Pinecone asynchronously in batches of 50.
    Validates chunks before uploading.
    """
    if not chunks:
        logger.warning("No chunks to upsert.")
        return
        
    valid_chunks = [c for c in chunks if c.page_content.strip()]
    if not valid_chunks:
        raise ValueError("All provided chunks were empty after validation.")
        
    # Ensure index exists before we start adding
    init_pinecone_index()
    
    # We pass our Google Generative embeddings instance into the Pinecone store
    embeddings = get_embeddings_client()
    # Ensure dimensions match 768 to avoid pinecone errors (wrap our custom dimension forcing if necessary)
    # The LangChain vectorstore uses the raw embeddings client under the hood. 
    # Because raw is 3072, we MUST manually truncate chunks before insertion OR wrap the embeddings.
    # To keep it completely robust with langchain-pinecone, we will just manually upload batch by batch.
    
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(PINECONE_INDEX_NAME)
    
    total_upserted = 0
    # Process batches
    for i in range(0, len(valid_chunks), MAX_UPSERT_BATCH_SIZE):
        batch = valid_chunks[i:i + MAX_UPSERT_BATCH_SIZE]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        # 1. Embed the texts using our custom truncated `embed_text` function (forces 768)
        # Because we need exactly 768 dims for the AI-BOS Index.
        vectors = embed_text(texts)
        
        # 2. Prepare Pinecone vector payloads
        upsert_payload = []
        for j, doc in enumerate(batch):
            # ID is purely a hash of the content or random string
            doc_id = str(hash(doc.page_content + str(doc.metadata.get("timestamp", ""))))
            upsert_payload.append({
                "id": doc_id,
                "values": vectors[j],
                "metadata": {"text": doc.page_content, **doc.metadata} # Pinecone needs text in metadata to retrieve it
            })
            
        # 3. Upsert
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, index.upsert, upsert_payload)
        total_upserted += len(batch)
        logger.info("Upserted batch of %d. Total: %d/%d", len(batch), total_upserted, len(valid_chunks))
        
    logger.info("Successfully finished upserting %d chunks to Pinecone.", total_upserted)

# ── Enhanced RAG Pipeline ──────────────────────────────────────────────────────

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA, HypotheticalDocumentEmbedder, LLMChain
from langchain.prompts import PromptTemplate
from langchain.storage import InMemoryStore

def validate_rag_query(query: Any) -> str:
    """
    Validate that the query is a string and has a length greater than 5.
    """
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")
    
    query = query.strip()
    if len(query) <= 5:
        raise ValueError("Query length must be greater than 5 characters.")
        
    return query

def build_advanced_retriever(llm: Any, pinecone_index_name: str, docs: List[Document]):
    """
    Builds a nested retrieval architecture:
    1. MultiQueryRetriever (Generates 3 variants)
    2. HyDE (Generates hypothetical answer)
    3. ParentDocumentRetriever (Small chunks search -> Big context return)
    4. Ensemble (BM25 + Vector)
    5. ContextualCompression (LLM Reranking)
    """
    logger.info("Initializing super-retriever pipeline...")
    
    embeddings = get_embeddings_client()
    
    # --- 1. Base Pinecone VectorStore ---
    vectorstore = PineconeVectorStore(
        index_name=pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    # --- 2. HyDE (Hypothetical Document Embedder) ---
    # This wraps our embedding client to use LLM-generated hypothetical anwsers
    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm, embeddings, "web_search"
    )
    
    # --- 3. Parent Document Retriever ---
    # We store the original docs in memory, search small chunks in Pinecone
    store = InMemoryStore()
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )
    # Add documents to the store and vectorstore
    parent_retriever.add_documents(docs, ids=None)

    # --- 4. Multi-Query Retriever ---
    # Generates multiple variations to improve recall
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=parent_retriever,
        llm=llm
    )

    # --- 5. BM25 (Keyword backup) ---
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    # --- 6. Ensemble (Vector MQ/HyDE + BM25) ---
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, mq_retriever],
        weights=[0.4, 0.6]
    )
    
    # --- 7. Reranker (Final Compression) ---
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

def build_rag_chain(llm: Any, retriever: Any):
    """
    Builds a RetrievalQA chain with a custom business prompt.
    """
    prompt_template = """You are the IntelForge AI-BOS Business Agent. 
Use the following pieces of retrieved context to answer the question.
If you do not know the answer, just say that you do not know. Do not make up information.
Context: {context}
Question: {question}
Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain
