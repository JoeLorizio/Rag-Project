import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
import time
t0 = time.time()
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
print(f"Imports loaded: {time.time() - t0:.2f}s")
t_start = time.time()

# Constants
CHROMA_DIR = "./chroma_store"

if not os.path.exists(CHROMA_DIR):
    raise RuntimeError("No vector store found. Run ingest.py first.")

# Load vector store
t0 = time.time()
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)
print(f"Vector store loaded: {time.time() - t0:.2f}s")

# Build chunk lookup for neighbor expansion
raw = vectorstore.get(include=["documents", "metadatas"])
chunk_lookup = {}
for doc, meta in zip(raw["documents"], raw["metadatas"]):
    chunk_lookup[meta["chunk_id"]] = {
        "text": doc,
        "source": meta["source"]
    }

# Reranker setup
t0 = time.time()
cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
print(f"Cross encoder loaded: {time.time() - t0:.2f}s")

reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 30})
)

# Set up the RAG chain
print("Setting up RAG chain...")
llm = OllamaLLM(model="llama3.2:3b", num_ctx=4096, temperature=0)

prompt = ChatPromptTemplate.from_template("""Answer using ONLY the context below. Be Short and Precise.
Context: {context}
                                          
Question: {question}""")

def expand_with_neighbors(chunks):
    seen = set()
    expanded = []

    for chunk in chunks:
        chunk_id = chunk.metadata["chunk_id"]
        source = chunk.metadata["source"]

        for neighbor_id in [chunk_id - 1, chunk_id, chunk_id + 1]:
            if neighbor_id in chunk_lookup:
                neighbor = chunk_lookup[neighbor_id]
                if neighbor["source"] == source and neighbor_id not in seen:
                    seen.add(neighbor_id)
                    expanded.append(neighbor["text"])

    return "\n\n".join(expanded)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": compression_retriever | expand_with_neighbors, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask a question

def ask(question: str, history: list = []):
    for token in chain.stream(question):
        yield token

if __name__ == "__main__":
    question = "List the engineers in the Nexus on-call rotation with their email addresses and phone numbers"
    print(f"\nQuestion: {question}")
    t_0 = time.time()
    for token in ask(question):
        print(token, end="", flush=True)
    print(f"\n\nTime to answer: {time.time() - t_0:.2f}s")
    print(f"\nTotal runtime after imports: {time.time() - t_start:.2f}s")
