import streamlit as st # Only needed for the cache decorator
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Optional

# Keep this import if FAISS is used, adjust if using a different vector store
from langchain.vectorstores.base import VectorStoreRetriever


def load_and_parse_document(uploaded_file) -> Optional[List[Document]]:
    """Loads and parses a document (PDF, TXT, DOCX) using Langchain loaders.

    Handles temporary file creation and cleanup.

    Args:
        uploaded_file: The file object uploaded via Streamlit.

    Returns:
        Optional[List[Document]]: A list of Langchain Document objects, or None on error.
    """
    docs = []
    tmp_file_path = None
    if uploaded_file is not None:
        try:
            # Create a temporary file to store the uploaded content.
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            # Select the appropriate loader based on the file extension.
            if file_extension == '.pdf':
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == '.txt':
                loader = TextLoader(tmp_file_path)
            elif file_extension == '.docx':
                loader = UnstructuredFileLoader(tmp_file_path)
            else:
                print(f"ERROR: Unsupported file type: {file_extension}")
                # Consider raising ValueError("Unsupported file type")
                return None

            docs = loader.load()
            print(f"DEBUG: Loaded {len(docs)} pages/docs from {uploaded_file.name}")
            return docs

        except Exception as e:
            print(f"ERROR: Error loading or parsing file {uploaded_file.name}: {e}")
            # Consider raising specific exceptions based on 'e'
            return None
        finally:
            # Ensure the temporary file is removed.
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    return None # Return None if uploaded_file was None initially

def chunk_documents(docs: List[Document], chunk_size=1000, chunk_overlap=200) -> Optional[List[Document]]:
    """Splits loaded documents into smaller chunks for vectorization.

    Args:
        docs (List[Document]): The list of documents loaded by Langchain.
        chunk_size (int): The target size for each chunk.
        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        Optional[List[Document]]: A list of chunked documents, or None on error.
    """
    if not docs:
        print("WARNING: No documents provided to chunk.")
        return None
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True, # Helps in potential debugging.
        )
        chunks = text_splitter.split_documents(docs)
        print(f"DEBUG: Split {len(docs)} document page(s) into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"ERROR: Error chunking documents: {e}")
        return None


@st.cache_resource # Keep cache for expensive model loading
def get_embeddings_model(model_name="all-MiniLM-L6-v2") -> Optional[SentenceTransformerEmbeddings]:
    """Loads the Sentence Transformer embeddings model, cached for efficiency.

    Args:
        model_name (str): The name of the Sentence Transformer model to load.

    Returns:
        Optional[SentenceTransformerEmbeddings]: The embeddings model instance or None on error.
    """
    print(f"INFO: Attempting to load embeddings model: {model_name}...")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        print("INFO: Embeddings model loaded successfully.")
        return embeddings
    except Exception as e:
        print(f"ERROR: Error loading embeddings model: {e}")
        # Provide hints for common installation issues
        if "sentence_transformers" in str(e):
             print("ERROR HINT: Please ensure 'sentence-transformers' is installed (`pip install sentence-transformers`)")
        elif "torch" in str(e):
             print("ERROR HINT: Please ensure 'torch' is installed (`pip install torch`)")
        return None


def create_vector_store_retriever(chunks: List[Document], embeddings: SentenceTransformerEmbeddings) -> Optional[VectorStoreRetriever]:
    """Creates a FAISS vector store from document chunks and returns a retriever.

    Args:
        chunks (List[Document]): The list of document chunks.
        embeddings (SentenceTransformerEmbeddings): The loaded embeddings model instance.

    Returns:
        Optional[VectorStoreRetriever]: The FAISS retriever instance or None on error.
    """
    if not chunks:
        print("WARNING: No document chunks found to create vector store.")
        return None
    if not embeddings:
        print("ERROR: Embeddings model not available. Cannot create vector store.")
        return None
    try:
        print(f"INFO: Creating vector store from {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("INFO: Vector store created successfully.")
        # Retrieve the top 3 most relevant chunks during search.
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"ERROR: Error creating FAISS vector store: {e}")
        if "faiss" in str(e):
            print("ERROR HINT: Please ensure 'faiss-cpu' or 'faiss-gpu' is installed (`pip install faiss-cpu`)")
        return None