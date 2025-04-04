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

# Load environment variables
load_dotenv()

PROMPT_TEMPLATE = """
You are an experienced pediatric respiratory disease specialist focusing on TB and pneumonia diagnosis, assisting another doctor in a conversational, analytical manner. 
Your goal is to systematically evaluate patient information and chat history using ONLY the provided reference materials to reach a well-supported diagnosis or recommendation.

IMPORTANT: You MUST cite the specific source documents in your response WITHOUT mentioning page numbers. For example, say "According to WHO guidelines..." or "Based on Uganda Clinical Guidelines..." or "From MamaOpe Clinical Manual...".

REFERENCE TEXT TO USE (cite exact sources in your response):
{context}

PATIENT'S CURRENT INFORMATION:
{patient_data}

PREVIOUS CONVERSATION:
{chat_history}

YOUR TASK:
1. ALWAYS START with an assessment of vital signs:
   - IMMEDIATELY ALERT the doctor if any life-threatening conditions exist.
   - Identify and flag any dangerous vital sign values for the patient.

2. Focus on diagnosing TB or pneumonia AFTER assessing vitals:
   - Analyze how the patient's signs and symptoms match diagnostic criteria for TB or pneumonia
   - Reference specific guidelines from the provided materials
   - Consider the patient's age and specific risk factors

3. Choose ONE action based on your analysis:

   IF VITAL SIGNS ARE DANGEROUS (regardless of other information):
   - Start with "ALERT: [description of critical condition]"
   - Recommend immediate interventions.
   
   IF INFORMATION IS INCOMPLETE (and vitals are not immediately dangerous):
   - Ask ONE clear, focused question to address the most important gap in diagnostic information
   - Explain why this information is needed, citing specific sources
   - Format as "Question: [your question]"

   IF INFORMATION IS SUFFICIENT:
   - Provide a detailed assessment with numbered findings from the reference materials
   - Recommend specific next steps (tests, treatments)
   - Format as "Assessment: [your assessment]"

RULES:
- ALWAYS start with vital sign assessment BEFORE asking any questions
- For pediatric patients, use age-appropriate vital sign ranges from reference materials
- Flag ANY dangerous condition IMMEDIATELY at the start of your response
- Always stay evidence-based and cite specific sources from the provided reference materials
- Focus primarily on TB and pneumonia diagnosis while acknowledging other serious possibilities
- Limit to 7 questions total across the conversation
- If 7 questions are reached, provide your best assessment with the available information
- Use plain, conversational language with clear clinical reasoning
"""

def is_diagnosis_complete(response: str) -> bool:
    response_lower = response.lower().strip()
    return "question:" not in response_lower    

async def generate_response(query: str, chat_history: str, patient_data: str, retriever):
    """Generate a diagnostic response using the LLM and retrieved context."""
    try:
        start_time = time.time()
        
        # Retrieve context
        context = retrieve_context(query, patient_data, retriever)
        print(f"Retrieved context: {context[:500]}...")

        # Count prior questions
        question_count = chat_history.count("Question:") if chat_history else 0
        if question_count >= 7:
            context += "\nNote: Maximum of 7 questions reached; provide an assessment with available data."
    
        # Populate prompt
        prompt = PROMPT_TEMPLATE.format(
            patient_data=patient_data,
            context=context,
            chat_history=chat_history or "No previous conversation",
        )

        # input_token_estimate = len(prompt) // 4
        # print(f"Estimated input tokens: {input_token_estimate}")

        # Generate model response with vertex AI
        model = GenerativeModel("gemini-1.5-pro-002")
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
        print(f"Unexpected error in generate_response: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    