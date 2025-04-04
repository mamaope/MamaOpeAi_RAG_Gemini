import os
import app.auth
import asyncio
from dotenv import load_dotenv
from app.services.vectorstore_manager import initialize_vectorstore, get_vectorstore
from app.services.vectordb_service import retrieve_context
from app.services.conversational_service import generate_response

load_dotenv()

async def test_vectorstore_and_model():
    try:
        print("\n=== Starting Vector Store and Model Test ===")
        
        # Initialize vector store
        print("Initializing vector store...")
        initialize_vectorstore()
        vectorstore = get_vectorstore()
        
        # Create a retriever
        print("Creating retriever...")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Test query
        test_query = "What are the symptoms of tuberculosis in children?"
        print(f"\nTest Query: {test_query}")
        test_patient_data = "Patient is a 5-year-old child with persistent cough for 3 weeks."
        
        # Get context from vector store
        print("\nRetrieving context from vector store...")
        context = retrieve_context(test_query, test_patient_data, retriever)
        
        # Print retrieved context
        print("\n=== Retrieved Context ===")
        print(context[:500] + "..." if len(context) > 500 else context)
        
        # Test model response
        print("\n=== Testing Model Response ===")
        response_text, diagnosis_complete = await generate_response(
            query=test_query,
            chat_history="",
            patient_data=test_patient_data,
            retriever=retriever
        )
        
        print("\nModel Response:")
        print(response_text)
        print(f"\nDiagnosis Complete: {diagnosis_complete}")
        
        # Verify if response contains information from the context
        print("\n=== Verifying Response ===")
        # Split the context by document sources
        doc_sources = []
        for line in context.split("\n"):
            if line.startswith("From "):
                source = line.replace("From ", "").replace(":", "")
                doc_sources.append(source)
        
        # Check if sources are cited in the response
        for source in set(doc_sources):
            if source in response_text:
                print(f"✅ Response contains information from {source}")
            else:
                print(f"❌ Response does not contain information from {source}")
        
    except Exception as e:
        print(f"\n❌ Error in test: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_vectorstore_and_model()) 
