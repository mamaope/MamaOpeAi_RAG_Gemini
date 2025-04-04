import app.auth
import langchain
print(f"LangChain version: {langchain.__version__}")
import asyncio
from app.services.vectorstore_manager import initialize_vectorstore, get_vectorstore
from app.services.vectordb_service import retrieve_context

async def test_retrieval():
    try:
        print("\n=== Testing Retrieval Logic ===")
        
        # Initialize vector store
        print("Initializing vector store...")
        initialize_vectorstore()
        vectorstore = get_vectorstore()
        
        # Create a retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 8, 
            }
        )
        
        # Test query (no patient data to focus purely on retrieval)
        test_query = "What are the symptoms of tuberculosis in children under 5 years old?"
        print(f"\nTest Query: {test_query}")
        
        # Retrieve context
        print("\nRetrieving context...")
        context = retrieve_context(test_query, "", retriever)  # Empty patient data
        
        # Analyze retrieved documents
        print("\n=== Retrieved Documents ===")
        documents = context.strip().split("\n\n") if context else []
        documents = [doc for doc in documents if doc.strip() and doc.startswith("From ")]
        print(f"Number of documents retrieved: {len(documents)} (expected: 8)")
        
        for i, doc in enumerate(documents, 1):
            print(f"\nDocument {i}:")
            print(doc[:200] + "..." if len(doc) > 200 else doc)
            print("-" * 50)
            
            # Check relevance based on key terms
            key_terms = ["tuberculosis", "tb", "symptoms", "signs", "children"]
            if any(term in doc.lower() for term in key_terms):
                print("✅ Relevant content found")
            else:
                print("❌ No relevant content found - possible retrieval issue")
        
        print("\n=== Simulated LLM Response ===")
        if documents:
            # Extract key information from each document
            response_lines = []
            for doc in documents:
                source = doc.split("\n")[0].replace("From ", "").strip()
                content = "\n".join(doc.split("\n")[1:]).strip()
                # Split content into sentences and take the first few relevant ones
                sentences = content.split(". ")
                relevant_sentences = []
                for sentence in sentences:
                    # Look for sentences that seem informative (e.g., contain key terms or numbers)
                    if any(term in sentence.lower() for term in ["tb", "tuberculosis", "children", "symptom", "sign"]) or any(char.isdigit() for char in sentence):
                        relevant_sentences.append(sentence.strip())
                    if len(relevant_sentences) >= 2:  # Limit to 2 sentences per document
                        break
                if relevant_sentences:
                    summary = ". ".join(relevant_sentences) + f". (Source: {source})"
                    response_lines.append(summary)
            
            if response_lines:
                response = "Based on the provided information: " + " ".join(response_lines)
            else:
                response = "Based on the provided information, no relevant details were found in the retrieved documents."
            print(response)
        else:
            print("No documents retrieved to generate a response.")
            
    except Exception as e:
        print(f"\n❌ Error in retrieval test: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
