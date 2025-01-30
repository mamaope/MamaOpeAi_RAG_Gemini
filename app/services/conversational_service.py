import google.generativeai as genai
import os
from app.services.vectordb_service import retrieve_context
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

PROMPT_TEMPLATE = """
You are an experienced respiratory disease specialist focusing on TB and pneumonia diagnosis. Your role is to assist another doctor by analyzing patient information using ONLY the provided reference materials.

REFERENCE TEXT TO USE:
{context}

PATIENT'S CURRENT INFORMATION:
{patient_data}

PREVIOUS CONVERSATION:
{chat_history}

YOUR TASK:
1. First analyze if previous questions were already answered in the patient information or chat history.
2. Then, choose ONE action:

   IF INFORMATION IS INCOMPLETE:
   - Ask exactly one clear, specific question about symptoms, medical history, or examination findings
   - The question must be something not already answered in the patient information or chat history
   - Explain briefly why you need this information based on the reference text

   IF INFORMATION IS SUFFICIENT:
   - Provide your preliminary assessment
   - List the specific findings from the reference text that support your assessment
   - Recommend next steps (tests or treatments) based solely on the reference text
   - Include a clear statement that this is to support, not replace, the doctor's clinical judgment

Remember:
- Never make suggestions that aren't supported by the reference text
- Don't ask for information that was already provided
- Maximum 6 questions before providing recommendations
- Stay focused on respiratory conditions, particularly TB and pneumonia
"""

def generate_response(query: str, chat_history: str, patient_data: str, retriever):
    try:
        # Retrieve context
        context = retrieve_context(query, retriever)
        # print(f"Retrieved context length: {len(context)}")
        # print(f"Context preview: {context[:200]}...")
        # print(f"Query: {query}")

        # Populate prompt
        prompt = PROMPT_TEMPLATE.format(
            patient_data=patient_data,
            context=context,
            chat_history=chat_history or "No previous conversation",
        )

        # Generate model response
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt).text

        # Check for diagnosis completion
        diagnosis_keywords = ["diagnosis:", "recommend:", "suggest:", "assessment:"]
        diagnosis_complete = any(keyword in response.lower() for keyword in diagnosis_keywords)
        
        return response, diagnosis_complete
    
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise

# def generate_response(query, retriever):
   
#     # Retrieve relevant context
#     context = retrieve_context(query, retriever)

#     # Populate the prompt template
#     prompt_template = """
#     You are an AI assistant. Answer the question below direclty using only the provided reference text.
#     **Do not include phrases such as "The provided text mentions" or "According to the text."**
#     If the reference text does not contain the necessary information, respond with "I cannot answer based on the provided information."

#     Reference Text:
#     {context}
#     =========
#     Question: {query}

#     Answer:
#     """
#     prompt = prompt_template.format(context=context, query=query)
#     print("Generated Prompt:")
#     print(prompt)
#     # Generate the response using the Gemini model
#     # model = genai.GenerativeModel("gemini-1.5-flash-latest")
#     model = genai.GenerativeModel("gemini-1.5-pro-latest")
#     response = model.generate_content(
#         prompt,
#         generation_config= genai.GenerationConfig(
#             temperature=0.1, 
#             top_k=10, 
#             top_p=0.8  
#         ),  
#     )

#     return response.text
