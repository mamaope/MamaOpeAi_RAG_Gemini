from app.services.vectordb_service import load_vectorstore_from_s3

vectorstore = None

def initialize_vectorstore():
    global vectorstore
    if not vectorstore:
        vectorstore = load_vectorstore_from_s3()
        if not vectorstore:
            raise RuntimeError("Failed to load vector store during initialization.")

def get_vectorstore():
    if not vectorstore:
        raise RuntimeError("Vector store is not initialized.")
    return vectorstore
