import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Define your constants
DB_DIR = "my_vector_db"
PDF_FILE = "Python Data Science Handbook - Jake VanderPlas.pdf"
COLLECTION_NAME = "vector_db"

# 1. Initialize the embedding model (required for both reading and writing)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Check if the database folder exists
if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    print("Database already exists. Skipping embedding and loading from disk...")

    # Load existing database
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
else:
    print("Database not found. Extracting PDF and creating vector embeddings...")

    # Load PDF
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()

    # Clean text to prevent Unicode errors
    for doc in docs:
        doc.page_content = doc.page_content.encode("utf-8", errors="ignore").decode("utf-8")

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    split_docs = splitter.split_documents(docs)

    # Embed and save to disk
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR
    )
    print("Database created and saved successfully!")

# ---------------------------------------------------------
# 3. RETRIEVAL PHASE
# At this point, `vector_db` is loaded regardless of which path the 'if' statement took.
# ---------------------------------------------------------

print("\n--- RAG System Ready ---")
question = "what is support vector machine?"
print(f"Searching for: '{question}'...")

retriver = vector_db.as_retriever(search_type = "mmr",
                                  search_kwargs = {"k":3,"lambda_mult":0.5})
result = retriver.invoke(question)

for i, res in enumerate(result):
    print(f"\n--- Result {i + 1} ---")
    print(res.page_content)