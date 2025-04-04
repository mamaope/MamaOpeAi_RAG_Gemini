from app.services.vectordb_service import create_vectorstore, load_vectorstore_from_s3
import app.auth

# Global vectorstore instance
vectorstore = None

def initialize_vectorstore():
    """Initialize the vector store by loading from S3 or creating if not found."""
    global vectorstore
    try:        
        if vectorstore is None:
            print("Initializing vectorstore...")
            vectorstore = load_vectorstore_from_s3()
            if not vectorstore:
                print("Creating new vectorstore...")
                create_vectorstore()
                vectorstore = load_vectorstore_from_s3()
            print("Vectorstore initialization complete")

    except Exception as e:
        print(f"Error during vector store initialization: {e}")
        import traceback
        print(traceback.format_exc())
        raise RuntimeError("Failed to initialize vector store.") from e            

def get_vectorstore():
    """Get the initialized vectorstore."""
    if not vectorstore:
        raise RuntimeError("Vector store is not initialized.")
    return vectorstore
