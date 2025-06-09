import json
import re
import tiktoken
from google.cloud import storage
from typing import List
from vertexai.language_models import TextEmbeddingModel

MODEL_MAX_TOKENS = 15000  # Reduced from 20000 for safety margin
SAFE_TARGET_CHUNK_TOKENS = 1000  # Smaller chunks to avoid quota issues
MIN_CONTENT_LENGTH = 30

# Don't initialize model here to avoid unnecessary API calls
# embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

def is_relevant_content(text: str) -> bool:
    """Filter out irrelevant content like references, footers, etc."""
    if len(text.strip()) < MIN_CONTENT_LENGTH:
        return False
    noisy_patterns = [
        r"^(References|Bibliography|Foreword|Preface|Acknowledgements|Acknowledgment|Index|Appendix)$",
        r"^(Page \d+ of \d+)$",
        r"^\d+\.\s+[A-Za-z]+.*\d{4};.*https?://doi\.org",
        r"^\d+\.\s+[A-Za-z]+.*\d{4};.*\d+:\d+",
        r"^[A-Za-z]+ [A-Za-z]+\..*\d{4};.*https?://doi\.org",
        r"^[A-Za-z]+ [A-Za-z]+\..*\d{4};.*\d+:\d+"
    ]
    for pattern in noisy_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    return True

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string."""
    if not text or not isinstance(text, str):
        return 0
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception as e:
        print(f"Token estimation error: {e}")
        # Fallback estimation: approx 4 chars per token
        return len(text) // 4

def clean_text(text: str) -> str:
    """Normalize and clean text for better chunking"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix broken sentences (period without space)
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove header/footer page numbers
    text = re.sub(r'\n+Page \d+ of \d+\n+', '\n', text)
    return text.strip()

def chunk_text(text: str, max_tokens: int = SAFE_TARGET_CHUNK_TOKENS, overlap_tokens: int = 50) -> List[str]:
    """Enhanced chunking strategy with smaller chunks and better handling of large texts"""
    if not text:
        return []

    # Clean the text first
    text = clean_text(text)
    
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    for para in paragraphs:
        # Skip very small paragraphs (likely noise)
        if len(para.strip()) < 20:
            continue
            
        para_tokens = estimate_tokens(para)
        
        # If paragraph itself is too large, split it by sentences
        if para_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para.strip())
            sent_chunk = ""
            sent_tokens = 0
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                sentence_tokens = estimate_tokens(sentence)
                
                # If single sentence is too large, split by words
                if sentence_tokens > max_tokens:
                    if sent_chunk:
                        chunks.append(sent_chunk)
                        sent_chunk = ""
                        sent_tokens = 0
                        
                    # Split large sentence by words
                    words = sentence.split()
                    word_chunk = ""
                    word_tokens = 0
                    
                    for word in words:
                        word_token = estimate_tokens(word + " ")
                        if word_tokens + word_token <= max_tokens:
                            word_chunk += word + " "
                            word_tokens += word_token
                        else:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word + " "
                            word_tokens = word_token
                            
                    if word_chunk:
                        chunks.append(word_chunk.strip())
                        
                elif sent_tokens + sentence_tokens <= max_tokens:
                    sent_chunk += " " + sentence if sent_chunk else sentence
                    sent_tokens += sentence_tokens
                else:
                    if sent_chunk:
                        chunks.append(sent_chunk)
                    sent_chunk = sentence
                    sent_tokens = sentence_tokens
                    
            if sent_chunk:
                chunks.append(sent_chunk)
                
        # If paragraph fits in current chunk, add it
        elif current_token_count + para_tokens <= max_tokens:
            separator = " " if current_chunk else ""
            current_chunk += separator + para
            current_token_count += para_tokens
        else:
            # Finalize current chunk and start a new one with this paragraph
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para
            current_token_count = para_tokens
            
    # Add the last chunk if any
    if current_chunk:
        chunks.append(current_chunk)
        
    # Verify all chunks are within limits
    final_chunks = []
    for chunk in chunks:
        token_count = estimate_tokens(chunk)
        if token_count > MODEL_MAX_TOKENS:
            print(f"WARNING: Chunk with {token_count} tokens exceeds limit - will be split further")
            # Emergency split by forcing smaller segments
            subchunks = chunk_text(chunk, max_tokens=max_tokens // 2)
            final_chunks.extend(subchunks)
        else:
            final_chunks.append(chunk)
            
    # Remove duplicates/near-duplicates
    deduplicated = []
    for chunk in final_chunks:
        # Skip very short chunks
        if len(chunk) < 100:
            continue
            
        # Check if this chunk is too similar to an existing one
        duplicate = False
        for existing in deduplicated:
            if chunk in existing or existing in chunk:
                overlap_percent = len(min(chunk, existing, key=len)) / len(max(chunk, existing, key=len))
                if overlap_percent > 0.7:  # 70% overlap is considered duplicate
                    duplicate = True
                    break
                    
        if not duplicate:
            deduplicated.append(chunk)
            
    print(f"Created {len(deduplicated)} chunks from document")
    return deduplicated

def fetch_processed_documents(bucket_name: str, prefix: str = "output/"):
    """Fetches processed JSON data from GCS, chunks text, and prepares metadata."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    texts = []
    metadatas = []
    doc_count = 0

    for blob in blobs:
        if blob.name.endswith("_extracted_text.json"):
            print(f"Processing document: {blob.name}")
            doc_count += 1
            try:
                data = json.loads(blob.download_as_text())
                text = data.get("text", "").strip()
                metadata = data.get("metadata", {})
                
                if not text or not is_relevant_content(text):
                    print(f"Skipping document - insufficient content: {blob.name}")
                    continue
                    
                token_estimate = estimate_tokens(text)
                if token_estimate > 500000:  # Very large document
                    print(f"WARNING: Very large document ({token_estimate} tokens): {blob.name}")
                    
                metadata.update({
                    "source": metadata.get("source", "Unknown"),
                    "content_length": len(text),
                    "filename": blob.name,
                })
                
                # Use smaller chunks to avoid quota issues
                chunks = chunk_text(text, max_tokens=SAFE_TARGET_CHUNK_TOKENS)
                if not chunks:
                    print(f"WARNING: No valid chunks created from {blob.name}")
                    continue
                    
                print(f"Created {len(chunks)} chunks from document {blob.name}")
                
                for i, chunk in enumerate(chunks):
                    chunk_tokens = estimate_tokens(chunk)
                    if chunk_tokens > MODEL_MAX_TOKENS:
                        print(f"ERROR: Chunk {i} has {chunk_tokens} tokens exceeding limit. Skipping.")
                        continue
                        
                    texts.append(chunk)
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": f"{doc_count}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "token_count": chunk_tokens
                    })
                    metadatas.append(chunk_metadata)
                    
            except Exception as e:
                print(f"Error processing document {blob.name}: {e}")
                
    print(f"Processed {doc_count} documents into {len(texts)} chunks for embedding")
    return texts, metadatas
