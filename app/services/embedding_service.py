from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from typing import List, Dict, Optional
from app.services.document_utils import estimate_tokens
import time
import random
import hashlib
import os
import json
from pathlib import Path

# Create cache directory
CACHE_DIR = Path("cache/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class EmbeddingFunction:
    def __init__(self):
        self._model = None
        self._cache = {}
        self._request_count = 0
        self._last_request_time = 0
        self._cache_hits = 0
        print("Initialized EmbeddingFunction with caching and rate limiting")

    @property
    def model(self):
        if self._model is None:
            print("Loading TextEmbeddingModel (this only happens once)")
            self._model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        return self._model

    def _get_cache_key(self, text: str):
        """Generate a cache key for the text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_from_cache(self, text: str):
        """Try to get embedding from cache"""
        cache_key = self._get_cache_key(text)
        
        # Check memory cache first
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
            
        # Then check file cache
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    embedding = json.load(f)
                    # Store in memory for faster future access
                    self._cache[cache_key] = embedding
                    self._cache_hits += 1
                    return embedding
            except Exception as e:
                print(f"Cache read error: {e}")
        
        return None

    def _save_to_cache(self, text: str, embedding: List[float]):
        """Save embedding to cache"""
        cache_key = self._get_cache_key(text)
        
        # Save to memory cache
        self._cache[cache_key] = embedding
        
        # Save to file cache
        cache_file = CACHE_DIR / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(embedding, f)
        except Exception as e:
            print(f"Cache write error: {e}")

    def _apply_rate_limit(self):
        """Apply rate limiting to avoid quota issues"""
        now = time.time()
        elapsed = now - self._last_request_time
        
        # Ensure at least 100ms between requests
        if elapsed < 0.1:
            sleep_time = 0.1 - elapsed + random.uniform(0, 0.05)
            time.sleep(sleep_time)
            
        # Add extra delay every 25 requests to avoid sustained quota limits
        if self._request_count > 0 and self._request_count % 25 == 0:
            time.sleep(1 + random.uniform(0, 0.5))
            
        self._last_request_time = time.time()
        self._request_count += 1

    def embed_query(self, text: str) -> List[float]:
        # Check cache first
        cached = self._get_from_cache(text)
        if cached:
            return cached
            
        # Apply rate limiting
        self._apply_rate_limit()
        
        # Try with exponential backoff
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                input_obj = TextEmbeddingInput(text, "RETRIEVAL_QUERY")
                embedding = self.model.get_embeddings([input_obj])[0].values
                
                # Save to cache
                self._save_to_cache(text, embedding)
                return embedding
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if "429" in error_msg or "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                    # Quota limit hit, back off exponentially
                    wait_time = (2 ** retry_count) + random.uniform(0, 1)
                    print(f"Quota limit hit. Retrying in {wait_time:.2f}s (attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    # Different error
                    print(f"Error during embed_query: {error_msg}")
                    if retry_count < max_retries:
                        time.sleep(1)
                    else:
                        raise
        
        raise RuntimeError(f"Failed to get embedding after {max_retries} retries")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
            
        # First check cache for all texts
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached:
                results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                
        if not uncached_texts:
            # All found in cache - sort by original index and return
            results.sort(key=lambda x: x[0])
            return [emb for _, emb in results]
            
        print(f"Cache hit rate: {len(results)}/{len(texts)} ({len(results)/len(texts):.0%})")
        
        # Create placeholder for all results
        all_embeddings = [None] * len(texts)
        
        # Fill in cached results
        for idx, emb in results:
            all_embeddings[idx] = emb
            
        # Process remaining texts in small batches
        inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in uncached_texts]
        batch_size = 10  # Start with a small batch size
        start_index = 0
        
        while start_index < len(uncached_texts):
            # Apply rate limiting
            self._apply_rate_limit()
            
            end_index = min(start_index + batch_size, len(uncached_texts))
            batch_inputs = inputs[start_index:end_index]
            batch_texts = uncached_texts[start_index:end_index]

            # Check total tokens in this batch
            total_tokens = sum(estimate_tokens(text) for text in batch_texts)
            
            # If too many tokens, reduce batch size
            while total_tokens > 8000 and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                end_index = min(start_index + batch_size, len(uncached_texts))
                batch_inputs = inputs[start_index:end_index]
                batch_texts = uncached_texts[start_index:end_index]
                total_tokens = sum(estimate_tokens(text) for text in batch_texts)

            print(f"Processing batch {start_index}-{end_index-1}: {len(batch_texts)} texts, ~{total_tokens} tokens")
            
            # Try with exponential backoff
            max_retries = 5
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    batch_embeddings = self.model.get_embeddings(batch_inputs)
                    success = True
                    
                    # Save results to cache and add to results array
                    for i, (text, embedding_obj) in enumerate(zip(batch_texts, batch_embeddings)):
                        embedding = embedding_obj.values
                        # Cache the embedding
                        self._save_to_cache(text, embedding)
                        # Add to final results
                        orig_idx = uncached_indices[start_index + i]
                        all_embeddings[orig_idx] = embedding
                        
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    if "429" in error_msg or "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                        # If we're hitting quota limits, reduce batch size
                        if batch_size > 1:
                            old_batch_size = batch_size
                            batch_size = max(1, batch_size // 2)
                            end_index = min(start_index + batch_size, len(uncached_texts))
                            batch_inputs = inputs[start_index:end_index]
                            batch_texts = uncached_texts[start_index:end_index]
                            print(f"Reducing batch size from {old_batch_size} to {batch_size} due to quota limits")
                        
                        # Back off exponentially
                        wait_time = (2 ** retry_count) + random.uniform(0, 1)
                        print(f"Quota limit hit. Retrying in {wait_time:.2f}s (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        # Different error
                        print(f"Error embedding batch: {error_msg}")
                        if retry_count < max_retries:
                            time.sleep(1)
                        else:
                            raise
            
            if not success:
                raise RuntimeError(f"Failed to embed batch after {max_retries} retries")
                
            # Move to next batch
            start_index = end_index
            
            # Small delay between batches
            time.sleep(0.5)
        
        # Verify all embeddings were generated
        if None in all_embeddings:
            raise RuntimeError("Some embeddings were not generated")
            
        return all_embeddings

    def validate_embedding(self, embedding: List[float]) -> bool:
        if not embedding:
            return False
        if len(embedding) != 768:
            return False
        if all(v == 0 for v in embedding):
            return False
        return True

    def get_stats(self):
        return {
            "cache_hits": self._cache_hits,
            "requests": self._request_count,
            "cache_size": len(self._cache)
        }

embed_fn = EmbeddingFunction()
