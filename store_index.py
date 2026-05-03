"""
Store Index — Build and populate the Pinecone vector store.

This script:
1. Loads all PDF and text documents from the data/ directory
2. Filters and cleans the content
3. Splits into chunks for embedding
4. Creates/uses a Pinecone index
5. Upserts all document embeddings

Usage:
    python store_index.py
"""

import os
import sys
from dotenv import load_dotenv

from src.helper import (
    load_all_documents,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not found in .env file.")
    print("   Please copy .env.example to .env and add your Pinecone API key.")
    sys.exit(1)

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in .env file.")
    print("   The chatbot feature will not work without it.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ---------------------------------------------------------------------------
# Load, process, and embed documents
# ---------------------------------------------------------------------------
print("📄 Loading documents from data/ ...")
extracted_data = load_all_documents("data")
print(f"   → Loaded {len(extracted_data)} raw document pages.")

print("🧹 Filtering and cleaning documents ...")
filtered_data = filter_to_minimal_docs(extracted_data)
print(f"   → {len(filtered_data)} pages after filtering.")

print("✂️  Splitting into chunks ...")
text_chunks = text_split(filtered_data)
print(f"   → {len(text_chunks)} text chunks created.")

print("🤖 Downloading HuggingFace embeddings model ...")
embeddings = download_hugging_face_embeddings()

# ---------------------------------------------------------------------------
# Create / populate Pinecone index
# ---------------------------------------------------------------------------
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

print("🔗 Connecting to Pinecone ...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    print(f"📦 Creating Pinecone index '{index_name}' ...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("   → Index created successfully.")
else:
    print(f"   → Index '{index_name}' already exists.")

print("📤 Uploading document embeddings to Pinecone ...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)

print("✅ Done! Vector store is ready.")