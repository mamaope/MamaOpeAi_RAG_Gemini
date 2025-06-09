from .vectordb_service import VectorStore, load_vectorstore

vectorstore = None

def initialize_vectorstore():
    global vectorstore
    try:
        if vectorstore is None:
            print("Initializing vectorstore...")
            vectorstore = load_vectorstore()
            print("Vectorstore initialization complete")
    except Exception as e:
        print(f"Error during vector store initialization: {e}")
        raise RuntimeError("Failed to initialize vector store.") from e

def get_vectorstore():
    if vectorstore is None:
        raise RuntimeError("Vector store is not initialized.")
    return vectorstore

initialize_vectorstore()
