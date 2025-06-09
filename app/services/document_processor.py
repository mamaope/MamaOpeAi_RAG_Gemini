import os
import json
import re
from google.cloud import documentai
from google.cloud import storage
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InternalServerError, RetryError
from typing import List
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_ID")
LOCATION = os.getenv("GCP_LOCATION")
PROCESSOR_ID = os.getenv("DOCUMENT_AI_PROCESSOR_ID")
GCS_BUCKET = os.getenv("GCS_BUCKET")
PROCESSOR_LOCATION = os.getenv("DOCUMENT_AI_PROCESSOR_LOCATION")

def batch_process_documents(
    project_id: str,
    location: str,
    processor_id: str,
    gcs_input_prefix: str,
    gcs_output_prefix: str,
    timeout: int = 3600,  # Increased timeout for large PDFs
):
    """Process all PDFs in the GCS directory using Document AI batch processing."""
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    name = client.processor_path(project_id, location, processor_id)

    gcs_prefix = documentai.GcsPrefix(gcs_uri_prefix=gcs_input_prefix)
    input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_prefix)

    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
        gcs_uri=gcs_output_prefix
    )
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)

    # Create the batch processing request
    request = documentai.BatchProcessRequest(
        name=name,
        input_documents=input_config,
        document_output_config=output_config,
    )

    # Start the batch processing operation
    operation = client.batch_process_documents(request)
    print(f"Started batch processing operation: {operation.operation.name}")

    # Wait for the operation to complete
    try:
        print(f"Waiting for operation {operation.operation.name} to complete...")
        operation.result(timeout=timeout)
    except (RetryError, InternalServerError) as e:
        print(f"Batch processing failed: {str(e)}")
        raise

    # Check the operation status
    metadata = documentai.BatchProcessMetadata(operation.metadata)
    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise ValueError(f"Batch Process Failed: {metadata.state_message}")

    print("Batch processing completed successfully.")

def fetch_processed_documents(bucket_name: str, output_prefix: str = "output/") -> List[dict]:
    """Fetch the processed documents from GCS and extract text."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = storage_client.list_blobs(bucket_name, prefix=output_prefix)

    processed_docs = []
    for blob in blobs:
        if blob.content_type != "application/json":
            print(f"Skipping non-JSON file: {blob.name} - Mimetype: {blob.content_type}")
            continue

        print(f"\nFetching processed document: {blob.name}")
        document = documentai.Document.from_json(
            blob.download_as_bytes(), ignore_unknown_fields=True
        )

        matches = re.match(r"gs://.*?/output/.*?/(.*?)/", f"gs://{bucket_name}/{blob.name}")
        source_uri = matches.group(1) if matches else "Unknown"

        full_text = []

        # Extract text from pages
        if hasattr(document, 'pages'):
            for page_num, page in enumerate(document.pages):
                # Extract text from blocks
                block_texts = []
                if hasattr(page, 'blocks'):
                    for block in page.blocks:
                        block_text = ""
                        if hasattr(block, 'layout') and hasattr(block.layout, 'text_anchor'):
                            for segment in block.layout.text_anchor.text_segments:
                                start = int(segment.start_index) if segment.start_index else 0
                                end = int(segment.end_index)
                                block_text += document.text[start:end]
                        if block_text.strip():
                            block_texts.append(block_text.strip())
                if block_texts:
                    full_text.append(f"\nBlocks (Page {page_num}):\n" + "\n".join(block_texts))
                print(f"Page {page_num}: Extracted {len(block_texts)} blocks, {sum(len(t) for t in block_texts)} characters")

                # Extract text from paragraphs
                paragraph_texts = []
                if hasattr(page, 'paragraphs'):
                    for paragraph in page.paragraphs:
                        paragraph_text = ""
                        if hasattr(paragraph, 'layout') and hasattr(paragraph.layout, 'text_anchor'):
                            for segment in paragraph.layout.text_anchor.text_segments:
                                start = int(segment.start_index) if segment.start_index else 0
                                end = int(segment.end_index)
                                paragraph_text += document.text[start:end]
                        if paragraph_text.strip():
                            paragraph_texts.append(paragraph_text.strip())
                if paragraph_texts:
                    full_text.append(f"\nParagraphs (Page {page_num}):\n" + "\n".join(paragraph_texts))
                print(f"Page {page_num}: Extracted {len(paragraph_texts)} paragraphs, {sum(len(t) for t in paragraph_texts)} characters")

                # Extract text from lines
                line_texts = []
                if hasattr(page, 'lines'):
                    for line in page.lines:
                        line_text = ""
                        if hasattr(line, 'layout') and hasattr(line.layout, 'text_anchor'):
                            for segment in line.layout.text_anchor.text_segments:
                                start = int(segment.start_index) if segment.start_index else 0
                                end = int(segment.end_index)
                                line_text += document.text[start:end]
                        if line_text.strip():
                            line_texts.append(line_text.strip())
                if line_texts:
                    full_text.append(f"\nLines (Page {page_num}):\n" + "\n".join(line_texts))
                print(f"Page {page_num}: Extracted {len(line_texts)} lines, {sum(len(t) for t in line_texts)} characters")

                # Extract text from tables
                table_texts = []
                if hasattr(page, 'tables'):
                    for table in page.tables:
                        table_rows = []
                        for row in (table.header_rows + table.body_rows if table.header_rows else table.body_rows):
                            row_text = []
                            for cell in row.cells:
                                cell_text = ""
                                if hasattr(cell, 'layout') and hasattr(cell.layout, 'text_anchor'):
                                    for segment in cell.layout.text_anchor.text_segments:
                                        start = int(segment.start_index) if segment.start_index else 0
                                        end = int(segment.end_index)
                                        cell_text += document.text[start:end]
                                row_text.append(cell_text.strip())
                            if row_text:
                                table_rows.append(" | ".join(row_text))
                        if table_rows:
                            table_texts.append("\n".join(table_rows))
                if table_texts:
                    full_text.append(f"\nTable (Page {page_num}):\n" + "\n".join(table_texts))
                print(f"Page {page_num}: Extracted {len(table_texts)} tables, {sum(len(t) for t in table_texts)} characters")

        # Combine all extracted text into a single string
        extracted_text = "\n".join(full_text) if full_text else document.text
        print(f"Total extracted text for {blob.name}: {len(extracted_text)} characters")

        # Save the extracted text and metadata
        processed_docs.append({
            "text": extracted_text,
            "metadata": {
                "source": f"gs://{bucket_name}/input/{source_uri}",
                "character_count": len(extracted_text),
                "page_count": len(document.pages) if hasattr(document, 'pages') else 0
            }
        })

        # Save the extracted text to a new JSON file
        output_blob = bucket.blob(f"output/{source_uri}_extracted_text.json")
        output_blob.upload_from_string(
            json.dumps({
                "text": extracted_text,
                "metadata": {
                    "source": f"gs://{bucket_name}/input/{source_uri}",
                    "character_count": len(extracted_text),
                    "page_count": len(document.pages) if hasattr(document, 'pages') else 0
                }
            }),
            content_type="application/json"
        )

    return processed_docs

def check_if_already_processed(bucket_name: str, output_prefix: str = "output/") -> bool:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = storage_client.list_blobs(bucket_name, prefix=output_prefix)
    for blob in blobs:
        if blob.name.endswith("_extracted_text.json"):
            return True
    return False

def process_documents_in_batch(bucket_name: str, input_prefix: str = "input/", output_prefix: str = "output/"):
    """Process all PDFs in the input directory using batch processing and fetch the results."""
    if check_if_already_processed(bucket_name, output_prefix):
        print("Documents already processed. Skipping batch processing.")
        processed_docs = fetch_processed_documents(bucket_name, output_prefix)
        print(f"Processed {len(processed_docs)} documents.")
        return
    
    gcs_input_prefix = f"gs://{bucket_name}/{input_prefix}"
    gcs_output_prefix = f"gs://{bucket_name}/{output_prefix}"
    
    batch_process_documents(
        project_id=PROJECT_ID,
        location=PROCESSOR_LOCATION,
        processor_id=PROCESSOR_ID,
        gcs_input_prefix=gcs_input_prefix,
        gcs_output_prefix=gcs_output_prefix,
        timeout=3600  # 1 hour timeout for large PDFs
    )

    # Step 2: Fetch the processed documents
    processed_docs = fetch_processed_documents(bucket_name, output_prefix)
    print(f"Processed {len(processed_docs)} documents.")

if __name__ == "__main__":
    process_documents_in_batch(bucket_name=GCS_BUCKET)
    