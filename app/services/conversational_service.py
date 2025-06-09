import os
import json
import asyncio
import time
from fastapi import HTTPException
from vertexai.generative_models import GenerativeModel, GenerationConfig
from app.services.vectordb_service import retrieve_context
import app.auth
from dotenv import load_dotenv
from typing import Dict
from google.api_core import exceptions
from tenacity import retry, stop_after_attempt, wait_exponential
from .vectordb_service import VectorSearchRetriever
from .vector_store_manager import get_vectorstore

# Load environment variables
load_dotenv()

PROMPT_TEMPLATE = """
You are an experienced doctor with a focus on respiratory diseases (e.g., TB and pneumonia) but with expertise in evaluating a wide range of conditions. 
Your role is assist another doctor by systematically evaluating patient information and chat history using ONLY the provided reference materials to reach 
a well-supported diagnosis or recommendation.

IMPORTANT: Follow all rules provided in the RULES section below without exception.

**Relevant Medical Guidelines and Context:**
{context}

**Patient Data:**
{patient_data}

**Conversation History:**
{chat_history}

**YOUR TASK:**
1. ALWAYS START with an assessment of vital signs:
   - IMMEDIATELY ALERT the doctor if any life-threatening conditions exist (e.g., heart rate, respiratory rate, oxygen saturation, temperature, or blood pressure outside safe ranges).
   - Do NOT repeat the vital sign values in your response unless they are dangerous and require an alert. 
   - If vital signs are dangerous, issue an alert and recommend immediate interventions without proceeding further. This takes priority over all other steps.

2. If vital signs are NOT life-threatenig, evaluate the patient's signs and symptoms:
   - Analyze how the patient's signs and symptoms match diagnostic criteria for TB or pneumonia.
   - **Do not be rigid:** if the signs and symptoms suggest other conditions (e.g., malaria, asthma, or any other disease), identify those as well.
   - Consider the patient's age, specific risk factors, and chat history to contextualize your evaluation.

3. Choose ONE action based on your analysis:

   IF VITAL SIGNS ARE DANGEROUS:
   - Start with "ALERT: [description of critical condition]"
   - List immediate interventions in a concise bullet list.
   - Do NOT ask questions or provide a diagnosis.
   
   IF YOU NEED MORE INFORMATION (e.g., missing key details like duration of symptoms, severity, etc):
   - Review the chat history to avoid asking redundant questions.
   - YOU MUST ask ONE clear, focused question to address the most important gap in diagnostic information.
   - Format your response as:
     - **Question:** [One-line focused question]
     - **Rationale:** [Numbered list of concise reasons why the question is critical]
   - Do NOT provide an impression or further management if you are asking a question.  

   IF INFORMATION IS SUFFICIENT AND YOU DONT NEED TO ASK A QUESTION (e.g., you have enough details to make an impression or suggest futher management):
   - Provide a concise assessment and recommendations.
   - Format your response as:
         - **Impression:** [Bullet list of key impressions with a concise reason for each, max 5 bullets] 
         - **Further Management:** [Bullet list of next steps, tests, treatments, or referrals, max 5 bullets]
         - *This application is designed to provide supportive health information and should be used only under the guidance of a qualified doctor or healthcare provider.*

**RULES:**
- For pediatric patients, use age-appropriate vital sign ranges from reference materials.
- Flag ANY dangerous condition IMMEDIATELY at the start of your response.
- Always remain evidence-based but DO NOT include any citations, references, or page numbers in your final output.
- While your primary focus is on respiratory conditions specifially TB and Pneumonia, do not ignore signs that may indicate other serious illnesses.
- Limit to 10 questions total across the conversation; track this by reviewing the chat history for previous "Question:" entries.
- If 10 questions are reached, provide your best assessment with the available information and do not ask futher questions.
- Use plain, conversational language with clear clinical reasoning.
- Keep responses short and focused: avoid lengthy explanations, and limit explanations to maximum 5 concise points.
- Only include the disclaimer (in italics) when providing an Impression and Further Management.

**Query:**
{query}

**Response:**
"""

def is_diagnosis_complete(response: str) -> bool:
    response_lower = response.lower().strip()
    return "question:" not in response_lower    

async def retrieve_context(query: str, patient_data: str, retriever: VectorSearchRetriever) -> str:
    enhanced_query = f"{query} {patient_data}".strip() if patient_data else query
    try:
        relevant_documents = retriever.invoke(enhanced_query)
        if not relevant_documents:
            return "No relevant documents found."
        
        contexts = []
        for doc in relevant_documents:
            source = doc.metadata.get('source', 'Unknown source').replace('.pdf', '')
            content = doc.page_content
            contexts.append(f"From {source}:\n{content}\n")
        
        return "\n".join(contexts)
    except Exception as e:
        print(f"Retrieval error: {e}")
        return f"An error occurred during retrieval: {str(e)}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_response(query: str, chat_history: str, patient_data: str, retriever: VectorSearchRetriever):
    """Generate a diagnostic response using the LLM and retrieved context."""
    try:
        start_time = time.time()
                
        context_start = time.time()
        context = await retrieve_context(query, patient_data, retriever)
        retrieval_time = time.time() - context_start
        
        print(f"Retrieved context for query '{query}': {context[:500]}...")

        # Count prior questions
        question_count = chat_history.count("Question:") if chat_history else 0
        if question_count >= 10:
            context += "\nNote: Maximum of 10 questions reached; provide an assessment with available data."
        
        # Populate prompt
        prompt = PROMPT_TEMPLATE.format(
            patient_data=patient_data,
            context=context,
            chat_history=chat_history or "No previous conversation",
            query=query
        )

        # Generate model response with vertex AI
        model = GenerativeModel("gemini-1.5-pro-002")
        generation_config = GenerationConfig(
            temperature=0.4,
            max_output_tokens=3000,
            top_p=0.9,
        )
        inference_start = time.time()
        response = await model.generate_content_async(prompt, generation_config=generation_config)
        inference_time = time.time() - inference_start

        response_text = response.text
        diagnosis_complete = is_diagnosis_complete(response_text)
        
        total_latency = time.time() - start_time
        print(f"Retrieval: {retrieval_time:.2f}s, Inference: {inference_time:.2f}s, Total: {total_latency:.2f}s")

        return response_text, diagnosis_complete
    
    except exceptions.ResourceExhausted:
        raise HTTPException(status_code=429, detail="Rate limit exceeded or resource unavailable, please try again later.")
    except exceptions.GoogleAPIError as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in generate_response: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    