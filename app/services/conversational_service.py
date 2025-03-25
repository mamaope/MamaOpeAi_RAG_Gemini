import os
import json
import asyncio
import time
from fastapi import HTTPException
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from app.services.vectordb_service import retrieve_context
from dotenv import load_dotenv
from typing import Dict
from google.api_core import exceptions

load_dotenv()

PROJECT_ID = os.getenv("GCP_ID")
PROJECT_LOCATION = os.getenv("GCP_LOCATION")

vertexai.init(project=PROJECT_ID, location=PROJECT_LOCATION)

PROMPT_TEMPLATE = """
You are an experienced respiratory disease specialist focusing on TB and pneumonia diagnosis, assisting another doctor in a conversational, analytical manner. 
Your goal is to systematically evaluate patient information and chat history using ONLY the provided reference materials to reach a well-supported diagnosis or recommendation.

REFERENCE TEXT TO USE (cite exact sources in your response):
{context}

PATIENT'S CURRENT INFORMATION:
{patient_data}

PREVIOUS CONVERSATION:
{chat_history}

YOUR TASK:
1. Review the patient’s current information and previous conversation to identify what’s already known—do not repeat questions already answered.
2. Analyze the signs and symptoms step by step, referencing the provided materials, and consider both TB and pneumonia as possibilities:
    - List what symptoms are present and how they align with TB or pneumonia criteria.
    - Identify critical missing information needed to confirm or rule out each condition.
3. Then, choose ONE action based on this analysis:

   IF INFORMATION IS INCOMPLETE:
   - Reflect on the current picture (e.g., 'The dry cough and difficulty breathing could suggest pneumonia, but we need more details to confirm').
   - Ask ONE clear, specific question about symptoms, medical history, or examination findings not yet provided, prioritising the most diagnostically significant gap.
   - Explain why this question is needed, citing the exact reference text (e.g., 'Uganda Clinical Guidelines requires fever duration for TB suspicion').
   - If asking a question, start with 'Question:'.

   IF INFORMATION IS SUFFICIENT:
   - Provide a detailed assessment.
   - List numbered findings from the reference text supporting your conclusion (e.g., '1. Persistent cough >2 weeks aligns with Uganda Clinical Guidelines').
   - Recommend next steps (e.g., tests or treatments) with citations to the reference text.

RULES:
- Cite specific reference text excerpts or sections in your response (e.g., name of the document, section)
- If asking a question, start with 'Question:'.
- Do not speculate or use external knowledge—stick to {context}.
- Limit to 7 questions total across conversations. If the limit is reached, provide a best-effort assessment or recommend consulting a specialist, starting with 'Assessment:'.
- Stay focused on TB and pneumonia, unless if signs and symptoms and chat history suggest a serious condition that needs urgent medical attention, then flag it.
- Use clear, concise, conversational plain text for your response, starting with either 'Question:' for a question or 'Assessment:' when providing a detailed assessment. Think aloud like a specialist, showing your reasoning.
"""

# Check for diagnosis completion
def is_diagnosis_complete(response: str) -> bool:
    response_lower = response.lower().strip()
    return "question:" not in response_lower    

async def generate_response(query: str, chat_history: str, patient_data: str, retriever):
    try:
        start_time = time.time()
        # Retrieve context
        context = retrieve_context(query, retriever)

        # Count prior questions
        question_count = chat_history.count("Question:") if chat_history else 0
        if question_count >= 7:
            context += "\nNote: Maximum of 7 questions reached; provide an assessment with available data or refer the doctor to a more experienced colleague."
    
        # Populate prompt
        prompt = PROMPT_TEMPLATE.format(
            patient_data=patient_data,
            context=context,
            chat_history=chat_history or "No previous conversation",
        )

        # input_token_estimate = len(prompt) // 4
        # print(f"Estimated input tokens: {input_token_estimate}")

        # Generate model response with vertex AI
        model = GenerativeModel("gemini-2.0-flash-001")
        generation_config = GenerationConfig(
            temperature=0.4,
            max_output_tokens=3000,
            top_p=0.9,
        )
        response = await model.generate_content_async(prompt, generation_config=generation_config)
        response_text = response.text

        # output_token_estimate = len(response_text) // 4
        # print(f"Estimated output tokens: {output_token_estimate}")
        diagnosis_complete = is_diagnosis_complete(response_text)
        
        # latency = time.time() - start_time
        # print(f"Latency: {latency:.2f}s")

        return response_text, diagnosis_complete
    
    except exceptions.ResourceExhausted:
        raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later.")
    except exceptions.GoogleAPIError as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    