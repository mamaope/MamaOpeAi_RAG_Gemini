import google.generativeai as genai
import os
from app.services.vectordb_service import retrieve_context
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

PROMPT_TEMPLATE = """
You are respiratory disease specialist specializing in respiratory illnesses particulary TB, pneumonia, or other conditions. 
Your role is to assist doctors by analyzing information and providing evidence-based insights STRICTLY from the provided reference materials.

REFERENCE KNOWLEDGE:
{context}

PATIENT INFORMATION:
{patient_data}

CONVERSATION HISTORY:
{chat_history}

INTERACTION RULES:

Based on the patient data and reference text:
- Rely strictly on the reference text for your response.
- If more information is needed, ask one leading question to gather relevant details.
- If you have enough information, provide a diagnosis (TB, pneumonia, or other conditions) and suggest further tests or treatments.

1. EVIDENCE BASIS
- Only use information explicitly stated in the reference text
- If suggesting something not in references, clearly mark it as "Unable to confirm from references"
- If more information is needed, ask one leading question to gather relevant details.

2. DIAGNOSTIC PROCESS
- Maximum 6 questions before providing recommendations, or diagnosis. Don't number the questions just ask them as though asking someone face to face.

3. DECISION POINTS
- After each answer, evaluate:
  * If sufficient information → Proceed to recommendations
  * If insufficient → Ask ONE specific question
  * If uncertain → Clearly state limitations

4. FINAL RECOMMENDATIONS OR PRELIMINARY DIAGNOSIS
- Must include:
  * Evidence-based findings (with reference text support)
  * Confidence level (High/Medium/Low)
  * Clear disclaimer about clinical judgment
  * Specific recommendations for additional tests/consultations

Remember: You are a support tool. Final clinical decisions rest with the healthcare provider.
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
