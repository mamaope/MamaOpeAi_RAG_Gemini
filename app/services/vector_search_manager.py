import os
import numpy as np
import time
import random
from typing import List
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()

VECTOR_SEARCH_INDEX_ID = os.getenv("VECTOR_SEARCH_INDEX_ID")
VECTOR_SEARCH_ENDPOINT_ID = os.getenv("VECTOR_SEARCH_ENDPOINT_ID")
DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID")

class VectorSearchManager:
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        self.index_id = VECTOR_SEARCH_INDEX_ID
        self.endpoint_id = VECTOR_SEARCH_ENDPOINT_ID
        self.deployed_index_id = DEPLOYED_INDEX_ID
        self.index_resource_name = f"projects/{self.project_id}/locations/{self.location}/indexes/{self.index_id}"
        self.index_endpoint_name = f"projects/{self.project_id}/locations/{self.location}/indexEndpoints/{self.endpoint_id}"
        self.index = None
        self.index_endpoint = None
        self.initialized = False
        print(f"VectorSearchManager initialized with project={project_id}, location={location}")
        print(f"Using index ID: {self.index_id}, endpoint ID: {self.endpoint_id}")

    def initialize(self):
        """Initialize the VertexAI API clients with retry logic"""
        if self.initialized:
            return
            
        print("Initializing Vector Search API clients...")
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                aiplatform.init(project=self.project_id, location=self.location)
                
                # Initialize index
                print(f"Initializing MatchingEngineIndex with name: {self.index_resource_name}")
                self.index = aiplatform.MatchingEngineIndex(index_name=self.index_resource_name)
                
                # Initialize endpoint
                print(f"Initializing MatchingEngineIndexEndpoint with name: {self.index_endpoint_name}")
                self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=self.index_endpoint_name
                )
                
                # Verify deployed index information
                if self.deployed_index_id is None:
                    deployed_indexes = self.index_endpoint.deployed_indexes
                    if not deployed_indexes:
                        print("WARNING: No deployed indexes found on this endpoint!")
                    else:
                        self.deployed_index_id = deployed_indexes[0].id
                        print(f"Using first deployed index ID: {self.deployed_index_id}")
                        
                self.initialized = True
                print("Vector Search API clients initialized successfully")
                return
                
            except Exception as e:
                retry_count += 1
                error_str = str(e)
                print(f"Error initializing Vector Search API: {error_str}")
                
                if retry_count < max_retries:
                    wait_time = (2 ** retry_count) + random.uniform(0, 1)
                    print(f"Retrying initialization in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to initialize Vector Search after {max_retries} attempts")
                    raise

    def upload_embeddings(self, embeddings: List[List[float]], ids: List[str]):
        """Upload embeddings to the vector search index with error handling and retries"""
        if not self.initialized:
            self.initialize()
            
        if len(embeddings) != len(ids):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of ids ({len(ids)})")
            
        if not embeddings:
            print("No embeddings to upload")
            return
            
        # Process in batches to avoid quota issues
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            print(f"Preparing batch {i//batch_size + 1}/{(len(embeddings) + batch_size - 1)//batch_size}: {len(batch_embeddings)} embeddings")
            
            datapoints = []
            for id_val, embedding in zip(batch_ids, batch_embeddings):
                datapoints.append({
                    "datapoint_id": str(id_val),
                    "feature_vector": embedding
                })
                
            # Upload with retries
            max_retries = 5
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    print(f"Uploading batch of {len(datapoints)} datapoints (attempt {retry_count + 1}/{max_retries})")
                    self.index.upsert_datapoints(datapoints=datapoints)
                    success = True
                    total_uploaded += len(datapoints)
                    print(f"Batch upload successful. Progress: {total_uploaded}/{len(embeddings)}")
                    
                except Exception as e:
                    retry_count += 1
                    error_str = str(e)
                    print(f"Error uploading batch: {error_str}")
                    
                    if "429" in error_str or "quota" in error_str.lower() or "limit" in error_str.lower():
                        # If hitting quota limits, reduce batch size
                        if batch_size > 10:
                            old_batch_size = batch_size
                            batch_size = max(10, batch_size // 2)
                            print(f"Reducing batch size from {old_batch_size} to {batch_size}")
                            
                            # Recalculate current batch
                            batch_end = min(i + batch_size, len(embeddings))
                            batch_embeddings = embeddings[i:batch_end]
                            batch_ids = ids[i:batch_end]
                            
                            # Rebuild datapoints
                            datapoints = []
                            for id_val, embedding in zip(batch_ids, batch_embeddings):
                                datapoints.append({
                                    "datapoint_id": str(id_val),
                                    "feature_vector": embedding
                                })
                    
                    # Apply exponential backoff
                    if retry_count < max_retries:
                        wait_time = (2 ** retry_count) + random.uniform(0, 1)
                        print(f"Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to upload batch after {max_retries} attempts")
                        raise
            
            # Add delay between batches to avoid rate limits
            if i + batch_size < len(embeddings):
                time.sleep(2)
                
        print(f"Vector upload complete: {total_uploaded}/{len(embeddings)} embeddings uploaded")

    def search(self, query_embedding: List[float], top_k: int = 5):
        """Search for similar vectors with error handling and retry logic"""
        if not self.initialized:
            self.initialize()
            
        if not self.deployed_index_id:
            raise ValueError("No deployed index ID available - cannot perform search")
            
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"Searching with deployed index: {self.deployed_index_id}")
                response = self.index_endpoint.find_neighbors(
                    deployed_index_id=self.deployed_index_id,
                    queries=[query_embedding],
                    num_neighbors=top_k
                )
                
                if not response or not response[0]:
                    print("Warning: Search returned no results")
                    return []
                    
                return response[0]
                
            except Exception as e:
                retry_count += 1
                error_str = str(e)
                print(f"Error during search: {error_str}")
                
                if retry_count < max_retries:
                    wait_time = (2 ** retry_count) + random.uniform(0, 1)
                    print(f"Retrying search in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to search after {max_retries} attempts")
                    if "not found" in error_str and "DeployedIndex" in error_str:
                        print(f"The deployed index ID '{self.deployed_index_id}' might be incorrect or missing")
                    raise
