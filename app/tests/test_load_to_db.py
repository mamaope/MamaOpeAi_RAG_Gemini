from app.services.vectordb_service import load_documents_to_db

def test_chromadb_operations():
    load_documents_to_db()

if __name__ == "__main__":
    test_chromadb_operations()