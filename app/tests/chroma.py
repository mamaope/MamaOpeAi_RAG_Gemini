import chromadb

# Create a ChromaDB client
client = chromadb.Client()

# Create a collection
collection = client.create_collection("my_collection")

# Add documents to the collection
collection.add(
    documents=["This is a document.", "Another document."],
    ids=["doc1", "doc2"]
)

# Query the collection
query_results = collection.query(query_texts=["document about something"])

print(query_results)
