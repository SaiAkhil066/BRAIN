import os
import time
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Start timer to measure total execution time
start_time = time.time()

# Verify CUDA availability and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Model and embedding setup with GPU support
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': device}  # Use GPU if available
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Load documents from the 'data/' directory
print("Loading PDF documents...")
loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()

# Adjust chunk size and overlap
context_token_limit = 512
overlap = 50
chunk_size = context_token_limit - overlap

# Split documents into chunks
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
texts = text_splitter.split_documents(documents)

# Create and persist the vector store
print("Creating vector store...")
vector_store = Chroma.from_documents(
    texts, 
    embeddings, 
    collection_metadata={"hnsw:space": "cosine"}, 
    persist_directory="stores/admin_cosine"
)

# End timer and calculate total execution time
end_time = time.time()
execution_time = end_time - start_time

print(f"Vector Store Created Successfully on {device.upper()}!")
print(f"Total Ingestion Time: {execution_time:.2f} seconds")
