import json
from app.services.vectordb_service import load_vectorstore_from_s3, retrieve_context, embed_fn

def test_retrieve_context():
    try:
        print("\n=== Starting Test ===")
        
        # Load vector store
        print("Loading vector store...")
        vectorstore = load_vectorstore_from_s3()
        if not vectorstore:
            print("Failed to load vector store.")
            return
        
        # Debug FAISS index
        print(f"FAISS index dimensionality: {vectorstore.index.d}")
        
        # Initialize retriever
        print("Initializing retriever...")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Test query
        query = "Tuberculosis in children"
        print(f"Test query: {query}")
        
        # Generate query embedding
        query_embedding = embed_fn.embed_query(query)
        print(f"Query embedding dimension: {len(query_embedding)}")
        
        # Validate dimensions
        if len(query_embedding) != vectorstore.index.d:
            raise ValueError(f"Query embedding dimension ({len(query_embedding)}) does not match FAISS index dimension ({vectorstore.index.d}).")
        
        # Retrieve context
        print("Retrieving context...")
        context = retrieve_context(query, retriever)
        
        # Output results
        print("\n=== Results ===")
        if "Error retrieving reference text:" in context:
            print("❌ Context retrieval failed")
            print(f"Error message: {context}")
        else:
            print("✅ Context retrieved successfully")
            print(f"Context length: {len(context)}")
            print("Context preview:")
            print(context[:300] + "...")
    
    except Exception as e:
        print(f"\n❌ Error in test function: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_retrieve_context()
