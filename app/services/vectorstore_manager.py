from app.services.vectordb_service import create_vectorstore, load_vectorstore_from_s3

vectorstore = None

def initialize_vectorstore():
    global vectorstore
    try:
        if vectorstore is None:
            vectorstore = load_vectorstore_from_s3()
            if not vectorstore:
                create_vectorstore()
                vectorstore = load_vectorstore_from_s3()

    except Exception as e:
        print(f"Error during vector store initialization: {e}")
        raise RuntimeError("Failed to initialize vector store.") from e            

def get_vectorstore():
    if not vectorstore:
        raise RuntimeError("Vector store is not initialized.")
    return vectorstore
