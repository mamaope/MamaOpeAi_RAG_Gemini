import json
import os
import google.generativeai as genai
import numpy as np
import boto3
import shutil
import tarfile
from google.api_core import retry
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List
from dotenv import load_dotenv

load_dotenv()

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
VECTORSTORE_S3_PREFIX = "output/vectorstore/"

class GeminiEmbeddingFunction(Embeddings):
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

def create_vectorstore():
    """Create FAISS vector store and upload to S3"""
    try:
        print("Loading documents...")
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file_path = os.path.join(root_dir, "documents.json")

        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        texts = [item["document"] for item in data]
        metadatas = [{"id": item["id"]} for item in data]

        # Create FAISS vector store
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embed_fn,
            metadatas=metadatas
        )

        # Upload vector store to S3
        print("Uploading vector store to S3...")
        upload_vectorstore_to_s3(vectorstore)
        print("Vector store created and uploaded successfully.")
    except Exception as e:
        print(f"Error creating vector store: {e}")

def upload_vectorstore_to_s3(vectorstore: FAISS):
    """Upload FAISS vector store files to S3."""
    try:
        s3 = boto3.client("s3")

        # Create a temporary directory to save vector store
        temp_dir = "/tmp/vectorstore"
        shutil.rmtree(temp_dir, ignore_errors=True)  # Clean up if it already exists
        os.makedirs(temp_dir, exist_ok=True)

        # Save vector store to the temporary directory
        vectorstore.save_local(temp_dir)

        # Compress the directory
        compressed_file = "/tmp/vectorstore.tar.gz"
        with tarfile.open(compressed_file, "w:gz") as tar:
            tar.add(temp_dir, arcname=".")

        # Upload the compressed file to S3
        s3.upload_file(
            Filename=compressed_file,
            Bucket=AWS_S3_BUCKET,
            Key=VECTORSTORE_S3_PREFIX + "vectorstore.tar.gz",
        )

        print("Vector store uploaded successfully to S3.")
    except Exception as e:
        print(f"Error uploading vector store to S3: {e}")

def load_vectorstore_from_s3():
    """Load the FAISS vector store directly from S3."""
    try:
        print("Loading vector store from S3...")
        s3 = boto3.client("s3")

        # Download the compressed vector store
        compressed_file = "/tmp/vectorstore.tar.gz"
        s3.download_file(
            Bucket=AWS_S3_BUCKET,
            Key=VECTORSTORE_S3_PREFIX + "vectorstore.tar.gz",
            Filename=compressed_file,
        )

        # Extract the compressed file
        temp_dir = "/tmp/vectorstore"
        shutil.rmtree(temp_dir, ignore_errors=True)  # Clean up if it already exists
        with tarfile.open(compressed_file, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # Load the vector store
        vectorstore = FAISS.load_local(temp_dir, embeddings=embed_fn, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully from S3.")
        return vectorstore
    except Exception as e:
        print(f"Error loading vector store from S3: {e}")
        return None
    
def retrieve_context(query, retriever):
    results = retriever.invoke(query)
    context = "\n".join([result.page_content for result in results])
    # print(f"Retrieved Context for Query '{query}':")
    # print(context)
    return context
 