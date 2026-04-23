import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DOCS_FOLDER = "docs"
CHROMA_DIR = "./chroma_store"

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} page(s)")
    return documents

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=60
)

print("Loading documents...")
documents = load_documents(DOCS_FOLDER)

print("Splitting into chunks...")
chunks = splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i

print(f"Created {len(chunks)} chunks")

print("Embedding and saving to Chroma...")
Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory=CHROMA_DIR
)

print("Done.")