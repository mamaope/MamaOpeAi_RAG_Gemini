import json
import os
import numpy as np
import boto3
import shutil
import tarfile
import faiss
import app.auth
import re
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from vertexai.language_models import TextEmbeddingModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()

# Initialize environment variables
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
VECTORSTORE_S3_PREFIX = "output/vectorstore/"
EMBEDDINGS_S3_PREFIX = "output/"

class EmbeddingFunction(Embeddings):
    """Embedding function using Azure OpenAI's text-embedding-3-large model since its the same
    embedding moodel that was used to create the document embeddings in Unstructured.io."""
    _client = None

    def __init__(self):
        pass

    @property
    def client(self):
        """Lazily load the Azure OpenAI client when first needed."""
        if EmbeddingFunction._client is None:
            try:
                print("Initializing Azure OpenAI client")
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                if not api_key or not endpoint:
                    raise ValueError("Azure OpenAI API key or endpoint not set.")
                EmbeddingFunction._client = AzureOpenAI(
                    api_version= os.getenv("API_VERSION"),  
                    azure_endpoint= endpoint,
                    api_key=api_key
                )
                print("Azure OpenAI client initialized successfully")
            except Exception as e:
                print(f"Error initializing Azure OpenAI client: {e}")
                raise
        return EmbeddingFunction._client

    def embed_query(self, text: str) -> List[float]:
        """Generate an embedding for a single query text."""
        embedding = self._embed([text])
        while isinstance(embedding, (list, tuple)) and embedding and isinstance(embedding[0], (list, tuple)):
            embedding = embedding[0]
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        embeddings = self._embed(texts)
        for i, embedding in enumerate(embeddings):
            while isinstance(embedding, (list, tuple)) and embedding and isinstance(embedding[0], (list, tuple)):
                print(f"Flattening nested embedding at index {i}: {embedding[:5]}...")
                embedding = embedding[0]
            embeddings[i] = embedding
        return embeddings

    def _embed(self, input: List[str]) -> List[List[float]]:
        """Core embedding logic using Azure OpenAI."""
        try:
            response = self.client.embeddings.create(
                input=input,
                model= os.getenv("MODEL_NAME")
            )
            
            # Get embeddings from response
            embeddings = [item.embedding for item in response.data]    
            return embeddings
        except Exception as e:
            print(f"Embedding error: {e}")
            raise
        
# Initialize the embedding function
embed_fn = EmbeddingFunction()

def is_relevant_content(text: str) -> bool:
    # Skip very short or empty content
    if len(text.strip()) < 30:
        return False    
    
    # patterns for noisy sections to exclude
    noisy_patterns = [
        r"^(References|Bibliography|Foreword|Preface|Acknowledgements|Acknowledgment|Index|Appendix)$",
        r"^(Page \d+ of \d+)$",
        r"^\d+\.\s+[A-Za-z]+.*\d{4};.*https?://doi\.org",  # Matches references with DOIs
        r"^\d+\.\s+[A-Za-z]+.*\d{4};.*\d+:\d+",  # Matches references with journal format (e.g., "2003;7:426–31")
        r"^[A-Za-z]+ [A-Za-z]+\..*\d{4};.*https?://doi\.org",  # Matches references starting with author names and DOIs
        r"^[A-Za-z]+ [A-Za-z]+\..*\d{4};.*\d+:\d+"  # Matches references starting with author names and journal format
    ]

    # Check if the text matches any noisy pattern
    for pattern in noisy_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False    
        
    return True

def create_vectorstore():
    """Create FAISS vector store using already-embedded data from S3."""
    try:
        print("Fetching processed files from S3...")
        s3 = boto3.client("s3")

        # List all JSON files in the output/ folder
        response = s3.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=EMBEDDINGS_S3_PREFIX)
        files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".json")]

        texts = []
        embeddings = []
        metadatas = []

        # Process each JSON file
        for file_key in files:
            obj = s3.get_object(Bucket=AWS_S3_BUCKET, Key=file_key)
            data = json.load(obj["Body"])

            for record in data:
                text = record.get("text", "").strip()
                metadata = record.get("metadata", {})
                embedding = record.get("embeddings")
                # Only include relevant content
                if text and embedding and is_relevant_content(text):
                    # Add more context to metadata
                    metadata.update({
                        "source": metadata.get("filename", "Unknown"),
                        "content_length": len(text),
                    })
                    texts.append(text)
                    metadatas.append(metadata)
                    # Ensure embedding is a list of floats
                    if isinstance(embedding, list):
                        embeddings.append(embedding)
                    else:
                        print(f"Warning: Skipping malformed embedding for text: {text[:50]}...")
                        continue

        if not texts or not embeddings:
            raise ValueError("No valid documents or embeddings found.")        

        # Convert embeddings to a NumPy array
        embeddings_np = np.array(embeddings, dtype=np.float32)
        # Validate the shape of the embeddings
        print(f"Embeddings shape: {embeddings_np.shape}")
        if len(embeddings_np.shape) != 2:
            raise ValueError(f"Invalid embeddings shape: {embeddings_np.shape}. Expected 2D array.")

        # Create FAISS index 
        dimension = embeddings_np.shape[1]  # Get embedding dimension
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)

        # Create sequential IDs and docstore
        docstore = {}
        index_to_docstore_id = {}
        
        for i in range(len(texts)):
            doc_id = f"doc_{i}"
            docstore[doc_id] = Document(
                page_content=texts[i],
                metadata=metadatas[i]
            )
            index_to_docstore_id[i] = doc_id

        # Create FAISS vector store
        print(f"Creating FAISS vector store with {len(docstore)} documents...")
        vectorstore = FAISS(
            embedding_function=embed_fn,
            index=index,
            docstore=InMemoryDocstore(docstore),
            index_to_docstore_id=index_to_docstore_id
        )

        # Save and upload vector store to S3
        print(f"Created vectorstore with {len(docstore)} documents")
        upload_vectorstore_to_s3(vectorstore)
        return vectorstore
    
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise

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
    
def retrieve_context(query: str, patient_data: str, retriever) -> str:
    """Retrieve relevant context for the query based on semantic similarity."""
    
    # Combine query with patient data
    enhanced_query = query
    if patient_data and patient_data.strip():
        enhanced_query = f"{query} {patient_data}"
        enhanced_query = ' '.join(enhanced_query.split())
    
    print(f"Enhanced query: {enhanced_query}")
    
    try:
        relevant_documents = retriever.invoke(enhanced_query)

        if not relevant_documents:
            return "No relevant documents found."
        
        # Post-retrieval filtering for relevance
        filtered_documents = []
        for doc in relevant_documents:
            print(f"Checking relevance for document: {doc.page_content[:50]}...")
            if is_relevant_content(doc.page_content):
                filtered_documents.append(doc)
            else:
                print(f"Document filtered out: {doc.page_content[:50]}...")
        
        if not filtered_documents:
            return "No relevant documents found after filtering."
        
        # Combine results with source tracking
        contexts = []
        for doc in filtered_documents:
            source = doc.metadata.get('filename', 'Unknown source')
            source = source.replace('.pdf', '')
            content = doc.page_content
            contexts.append(f"From {source}:\n{content}\n")

        return "\n".join(contexts)
    except Exception as e:
        print(f"Retrieval error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return f"An error occurred during retrieval: {str(e)}"
    
