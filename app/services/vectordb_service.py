import json
import os
import google.generativeai as genai

from google.api_core import retry
from langchain_community.vectorstores import FAISS
from typing import List

class GeminiEmbeddingFunction():
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # print(f"Embedding documents (first 5): {texts[:5]}")  
        self.document_mode = True 
        return self.__call__(texts)

    def embed_query(self, text: str) -> List[float]:
        self.document_mode = False 
        return self.__call__([text])[0]

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Embedding logic using Gemini's model
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        embeddings = response["embedding"]
        return embeddings

embed_fn = GeminiEmbeddingFunction()

# Load documents and IDs into FAISS
def load_documents_to_db():

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(root_dir, "documents.json")

    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        texts = [item["document"] for item in data]
        metadatas = [{"id": item["id"]} for item in data] 

        # print(f"Sample Texts: {texts[:3]}") 
        # print(f"Sample Metadata: {metadatas[:3]}")  
        # Create FAISS vector store
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embed_fn,
            metadatas=metadatas
        )
        print("FAISS vector store created successfully!")
        return vectorstore
    except Exception as e:
        print(f"Error during FAISS DB operations: {str(e)}")    

def retrieve_context(query, retriever):
    embed_fn.document_mode = False
    results = retriever.get_relevant_documents(query)
    context = "\n".join([result.page_content for result in results])
    # print(f"Retrieved Context for Query '{query}':")
    # print(context)
    return context
 