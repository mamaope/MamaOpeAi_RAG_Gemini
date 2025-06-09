import json
import os
import numpy as np
import app.auth
import re
import time
import random
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from typing import List, Optional
from dotenv import load_dotenv
from .vector_search_manager import VectorSearchManager
from .embedding_service import embed_fn
from .document_utils import chunk_text, fetch_processed_documents, estimate_tokens

load_dotenv()

class VectorStore:
    def __init__(self, project_id: str, location: str):
        self.vector_search = VectorSearchManager(project_id=project_id, location=location)
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}
        self.embedding_function = embed_fn

    def create_and_upload(self, texts: List[str], metadatas: List[dict]):
        # Log some info about the texts
        print(f"Processing {len(texts)} text chunks for embedding and vector storage")
        
        # Validate text sizes before embedding
        max_token_limit = 15000  # Reduced from 20000 for safety
        valid_texts = []
        valid_metadatas = []
        
        for i, text in enumerate(texts):
            token_count = estimate_tokens(text)
            if token_count > max_token_limit:
                print(f"Skipping oversized text at index {i} ({token_count} tokens)")
                continue
            
            # Add valid texts to processing queue
            valid_texts.append(text)
            valid_metadatas.append(metadatas[i])
            
        print(f"After filtering, {len(valid_texts)}/{len(texts)} texts will be processed")

        # Embed in smaller batches with retry logic
        batch_size = 20  # Smaller batches to avoid quota issues
        embeddings = []
        
        for i in range(0, len(valid_texts), batch_size):
            # Get current batch
            end_idx = min(i + batch_size, len(valid_texts))
            batch_texts = valid_texts[i:end_idx]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(valid_texts) + batch_size - 1)//batch_size}: {len(batch_texts)} texts")
            
            # Try embedding with backoff
            max_retries = 5
            retry_count = 0
            batch_embeddings = None
            
            while retry_count < max_retries and batch_embeddings is None:
                try:
                    # Small delay between batches
                    if i > 0:
                        time.sleep(1)  # Pause between batches
                        
                    batch_embeddings = self.embedding_function.embed_documents(batch_texts)
                except Exception as e:
                    retry_count += 1
                    error_str = str(e)
                    
                    if "429" in error_str or "quota" in error_str.lower() or "limit" in error_str.lower():
                        # Rate limit hit - backoff exponentially
                        wait_time = (2 ** retry_count) + random.uniform(0, 1)
                        print(f"Rate limit hit, backing off for {wait_time:.1f}s (attempt {retry_count}/{max_retries})")
                        
                        # If multiple retries, reduce batch size
                        if retry_count > 2 and batch_size > 5:
                            old_size = batch_size
                            batch_size = max(5, batch_size // 2)
                            print(f"Reducing batch size from {old_size} to {batch_size}")
                            
                            # Recalculate batch with new size
                            end_idx = min(i + batch_size, len(valid_texts))
                            batch_texts = valid_texts[i:end_idx]
                        
                        time.sleep(wait_time)
                    else:
                        print(f"Error embedding batch: {error_str}")
                        if retry_count < max_retries:
                            time.sleep(2)
                        else:
                            raise
            
            # If we couldn't embed after max retries, skip batch
            if batch_embeddings is None:
                print(f"Failed to embed batch after {max_retries} retries, skipping")
                continue
                
            embeddings.extend(batch_embeddings)

        # Validate embeddings
        print(f"Validating {len(embeddings)} embeddings")
        valid_embeddings = []
        final_texts = []
        final_metadatas = []
        
        for i, (embedding, text, metadata) in enumerate(zip(embeddings, valid_texts, valid_metadatas)):
            if self.embedding_function.validate_embedding(embedding):
                valid_embeddings.append(embedding)
                final_texts.append(text)
                final_metadatas.append(metadata)
            else:
                print(f"Invalid embedding at index {i} - skipping")
        
        # Convert to numpy array
        if len(valid_embeddings) == 0:
            raise ValueError("No valid embeddings generated. Cannot proceed.")
            
        print(f"Preparing {len(valid_embeddings)} valid embeddings for upload")
        embeddings_np = np.array(valid_embeddings, dtype=np.float32)
        
        # Generate IDs and store documents
        ids = []
        for i, (text, metadata) in enumerate(zip(final_texts, final_metadatas)):
            doc_id = f"doc_{i}"
            self.docstore._dict[doc_id] = Document(page_content=text, metadata=metadata)
            self.index_to_docstore_id[i] = doc_id
            ids.append(doc_id)
        
        # Upload to vector store with retry logic
        max_upload_retries = 3
        upload_retry = 0
        upload_success = False
        
        while upload_retry < max_upload_retries and not upload_success:
            try:
                print(f"Uploading {len(ids)} embeddings to Vector Search (attempt {upload_retry+1}/{max_upload_retries})")
                self.vector_search.upload_embeddings(embeddings_np.tolist(), ids)
                upload_success = True
                print(f"Successfully uploaded {len(ids)} embeddings to Vector Search")
            except Exception as e:
                upload_retry += 1
                error_str = str(e)
                print(f"Error during upload: {error_str}")
                
                if upload_retry < max_upload_retries:
                    wait_time = (2 ** upload_retry) + random.uniform(0, 1)
                    print(f"Retrying upload in {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    print("Failed all upload attempts")
                    raise
        
        print("Vector store creation complete")

    def as_retriever(self, search_type: str = "similarity", search_kwargs: dict = None):
        return VectorSearchRetriever(self)
    
class VectorSearchRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.search_kwargs = {"k": 5} if search_kwargs is None else search_kwargs

    def invoke(self, query: str):
        # Apply retry logic for embedding the query
        max_retries = 3
        retry_count = 0
        query_embedding = None
        
        while retry_count < max_retries and query_embedding is None:
            try:
                query_embedding = self.vector_store.embedding_function.embed_query(query)
            except Exception as e:
                retry_count += 1
                error_str = str(e)
                
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = (2 ** retry_count) + random.uniform(0, 1)
                    print(f"Rate limit hit during query embedding. Retrying in {wait_time:.1f}s ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Error embedding query: {error_str}")
                    if retry_count < max_retries:
                        time.sleep(1)
                    else:
                        raise
        
        if query_embedding is None:
            raise RuntimeError(f"Failed to generate query embedding after {max_retries} attempts")
            
        if not self.vector_store.embedding_function.validate_embedding(query_embedding):
            raise ValueError("Invalid query embedding")
            
        # Apply retry logic for search
        search_retries = 3
        search_count = 0
        neighbors = None
        
        while search_count < search_retries and neighbors is None:
            try:
                neighbors = self.vector_store.vector_search.search(query_embedding, top_k=self.search_kwargs["k"])
            except Exception as e:
                search_count += 1
                error_str = str(e)
                print(f"Error during search: {error_str}")
                
                if search_count < search_retries:
                    wait_time = (2 ** search_count) + random.uniform(0, 1)
                    print(f"Retrying search in {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    raise
        
        if neighbors is None:
            raise RuntimeError(f"Failed to search after {search_retries} attempts")
            
        return [
            self.vector_store.docstore._dict[self.vector_store.index_to_docstore_id[int(neighbor.id)]]
            for neighbor in neighbors
        ]
    
def create_vectorstore():
    project_id = os.getenv("GCP_ID")
    location = os.getenv("GCP_LOCATION")
    bucket_name = os.getenv("GCS_BUCKET")
    
    print(f"Fetching and processing documents from bucket: {bucket_name}")
    texts, metadatas = fetch_processed_documents(bucket_name)
    
    if not texts:
        raise ValueError("No valid documents found or processed")
    
    print(f"Creating vector store with {len(texts)} text chunks")
    vectorstore = VectorStore(project_id=project_id, location=location)
    vectorstore.create_and_upload(texts, metadatas)
    return vectorstore    

def load_vectorstore():
    project_id = os.getenv("GCP_ID")
    location = os.getenv("GCP_LOCATION")
    return VectorStore(project_id=project_id, location=location)    

if __name__ == "__main__":
    try:
        print("Starting vector store creation process...")
        create_vectorstore()
        print("Vector store creation completed successfully")
    except Exception as e:
        print(f"ERROR during vector store creation: {str(e)}")
        raise
    