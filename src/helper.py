"""
Helper utilities for the Medical Agent RAG pipeline.

Provides functions for:
- Loading and parsing PDF/text documents
- Splitting documents into chunks for embedding
- Downloading HuggingFace embeddings
- Text cleaning and sanitization
"""

import os
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove invalid unicode, surrogates, and excessive whitespace."""
    if not isinstance(text, str):
        return str(text)
    # Remove unicode surrogates
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def load_pdf_files(data_dir: str) -> list:
    """
    Recursively load all PDF files from *data_dir*.
    Returns a flat list of LangChain Document objects.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()
    return documents


def load_text_files(data_dir: str) -> list:
    """
    Recursively load all .txt files from *data_dir*.
    Returns a flat list of LangChain Document objects.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    return documents


def load_all_documents(data_dir: str) -> list:
    """Load both PDF and TXT documents from *data_dir*."""
    docs = []
    docs.extend(load_pdf_files(data_dir))
    docs.extend(load_text_files(data_dir))
    return docs


# ---------------------------------------------------------------------------
# Filtering / Pre-processing
# ---------------------------------------------------------------------------

def filter_to_minimal_docs(documents: list) -> list:
    """
    Clean document content — strip whitespace, remove empty pages,
    and sanitize unicode issues.
    """
    filtered = []
    for doc in documents:
        content = clean_text(doc.page_content)
        if len(content) > 50:  # skip near-empty pages
            doc.page_content = content
            filtered.append(doc)
    return filtered


# ---------------------------------------------------------------------------
# Text Splitting
# ---------------------------------------------------------------------------

def text_split(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split documents into smaller chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Download and return a HuggingFace sentence-transformer embedding model.
    Default model produces 384-dimensional vectors.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
