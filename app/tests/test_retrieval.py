from app.services.conversational_service import generate_response
from app.services.vectordb_service import load_documents_to_db

if __name__ == "__main__":
    query = "What are the symptoms of tuberculosis in children less than 5 years?"

    try:
        # Load the FAISS vector store
        vectorstore = load_documents_to_db()
        if not vectorstore:
            raise Exception("Failed to load FAISS vector store.")

        # Initialize the retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
            
        )

        # Generate response
        answer = generate_response(query, retriever)

        # Display the result
        print("Query:", query)
        print("Answer:", answer)

    except Exception as e:
        print(f"Error generating response: {str(e)}")
