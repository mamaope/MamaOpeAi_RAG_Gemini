import google.generativeai as genai
import os
from app.services.vectordb_service import retrieve_context
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

PROMPT_TEMPLATE = """
You are an experienced doctor specializing in respiratory illnesses.
You role is to assist another doctor in diagnosing a patient by asking **specific and concise questions** based on the patient data and reference text.
**Strictly avoid phrases like "We need more information" or any other introductory statements. Only ask the next relevant question directly.**
If the reference text does not contain the necessary information, respond with "Sorry, I cannot answer based on the provided information."

Patient Data:
{patient_data}
=========
Reference Text:
{context}
=========
Chat History:
{chat_history}
=========
Guidelines:
- The question should be concise and relevant to the patient's symptoms and medical history.
- Example of a direct question: "Does the patient have weight loss or swollen lymph nodes?"

Response:
"""

def generate_response(query: str, chat_history: str, patient_data: Dict[str, str], retriever):

    # Retrieve context
    context = retrieve_context(query, retriever)

    # Populate prompt
    prompt = PROMPT_TEMPLATE.format(
        patient_data="\n".join(f"{key}: {value}" for key, value in patient_data.items()),
        context=context,
        chat_history=chat_history,
    )

    # print("Generated Prompt:")
    # print(prompt)
    # Generate model response
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            top_k=10,
            top_p=0.8
        ),    
    ).text

    # Check if the response is complete
    diagnosis_complete = any(
        keyword in response.lower()
        for keyword in ["diagnosis, recommendation", "suggest"]
    )
    return response, diagnosis_complete

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
