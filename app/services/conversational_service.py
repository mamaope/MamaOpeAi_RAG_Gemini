import os
import json
import asyncio
import time
import re
from fastapi import HTTPException
from vertexai.generative_models import GenerativeModel, GenerationConfig
from app.services.vectordb_service import retrieve_context
import app.auth
from dotenv import load_dotenv
from typing import Dict, Tuple, Any
from google.api_core import exceptions
from tenacity import retry, stop_after_attempt, wait_exponential

# # Load environment variables
# load_dotenv()

# # Prompt templates for different query types
# DIFFERENTIAL_DIAGNOSIS_PROMPT = """
# You are an expert medical AI, designed to assist a qualified doctor. Your task is to analyze patient information and provide a structured clinical assessment based ONLY on the provided reference materials.

# IMPORTANT: You must follow all rules listed below.

# ---
# RULES:
# 1.  **Source of Truth**: Base your entire analysis strictly on the `REFERENCE TEXT TO USE`. Do NOT reference sources that are not explicitly provided in the knowledge base.
# 2.  **Selective Citations**: Only cite sources when:
#     - Making specific clinical recommendations or diagnostic criteria
#     - Stating clinical guidelines or protocols
#     - Referencing specific diagnostic tests or treatment approaches
#     - Quoting or paraphrasing specific clinical information
#    Do NOT cite sources for general medical knowledge, basic pathophysiology, or common clinical observations.
# 3.  **Accurate Citations**: When citing, use ONLY the exact source names from the `AVAILABLE KNOWLEDGE BASE SOURCES` list. Include page numbers, section names, or specific document identifiers when available in the format: `[Source: document_name, page/section]`.
# 4.  **Medical Acronyms**: Interpret and correctly use common medical acronyms.
# 5.  **Critical Assessment**: Always begin by checking for life-threatening conditions. If any vital signs are in a dangerous range, issue an "CRITICAL ALERT" immediately and prioritize urgent interventions.
# 6.  **Output Format**: Structure your response using the following format:

# **CLINICAL OVERVIEW**
# [Provide a brief 1-2 paragraph overview summarizing the case in relation to the question or user query, highlighting key clinical features and initial impression]

# **MOST LIKELY DIAGNOSIS**
# [Present the primary diagnosis with brief supporting rationale]

# **DIFFERENTIAL DIAGNOSES**
# 1. **[Primary Diagnosis]**: [Brief explanation with supporting/opposing evidence, cite specific guidelines when applicable]
# 2. **[Secondary Diagnosis]**: [Brief explanation with supporting/opposing evidence, cite specific guidelines when applicable]  
# 3. **[Tertiary Diagnosis]**: [Brief explanation with supporting/opposing evidence, cite specific guidelines when applicable]
# [Add more if clinically relevant - aim for 3+ when possible]

# **⚡ IMMEDIATE WORKUP & INVESTIGATIONS**
# - [List essential immediate tests/investigations needed]
# - [Include time-sensitive protocols if applicable]
# - [Prioritize based on clinical urgency]

# **MANAGEMENT & RECOMMENDATIONS**
# - [Immediate management steps]
# - [Treatment recommendations with specific interventions]
# - [Monitoring requirements]
# - [Follow-up plans]

# **RED FLAGS / DANGER SIGNS**
# - [List warning signs that require immediate attention]
# - [Critical deterioration indicators]

# **ADDITIONAL INFORMATION NEEDED** (if applicable)
# [Only include this section if you need more critical information to refine the diagnosis or management plan. Ask one focused, essential question.]

# **Sources:** {sources}

# *This application is for clinical decision support and should only be used by qualified healthcare professionals.*

# ---

# REFERENCE TEXT TO USE:
# {context}

# AVAILABLE KNOWLEDGE BASE SOURCES:
# {sources}

# PATIENT'S CURRENT INFORMATION:
# {patient_data}

# PREVIOUS CONVERSATION:
# {chat_history}

# YOUR TASK:
# Evaluate the patient's information according to the rules and provide your response in the specified format. Use emojis as section headers as shown. Focus on practical, actionable clinical guidance. Remember: cite only when referencing specific clinical guidelines, protocols, or criteria from the provided sources. Do not cite general medical knowledge or make up source names.
# """

# DRUG_INFORMATION_PROMPT = """
# You are an expert pharmacology AI, designed to assist qualified healthcare professionals with drug information. Your task is to provide comprehensive drug information based ONLY on the provided reference materials from the drug knowledge base.

# IMPORTANT: You must follow all rules listed below.

# ---
# RULES:
# 1.  **Source of Truth**: Base your entire response strictly on the `REFERENCE TEXT TO USE`. Do NOT make up or infer information not explicitly provided in the knowledge base.
# 2.  **Accurate Information**: Only provide information that is directly found in the drug database. If specific information is not available, clearly state "Information not available in knowledge base."
# 3.  **Accurate Citations**: When citing, use ONLY the exact source names from the `AVAILABLE KNOWLEDGE BASE SOURCES` list. For drug database sources, include the full citation with URL when available.
# 4.  **Drug Names**: Use the exact drug names as they appear in the knowledge base.
# 5.  **Output Format**: Structure your response using the following format:

# **DRUG OVERVIEW**
# [Provide a brief overview of the drug based on the knowledge base, including what it is, active ingredients, and primary indications]

# **SIDE EFFECTS**
# [List side effects by frequency categories as found in the knowledge base:
# - Very Common (>1/10)
# - Common (1/100 to 1/10) 
# - Uncommon (1/1,000 to 1/100)
# - Rare (1/10,000 to 1/1,000)
# - Very Rare (<1/10,000)
# - Unknown frequency
# Only include categories that have documented side effects in the knowledge base]

# **DRUG INTERACTIONS**
# [List known drug interactions from the knowledge base. If none are documented, state "No drug interactions documented in knowledge base."]

# **CONTRAINDICATIONS**
# [List contraindications from the knowledge base. If none are documented, state "No contraindications documented in knowledge base."]

# **MECHANISM OF ACTION** (if available)
# [Include molecular targets, mechanisms of action, and pharmacological information if available in the knowledge base]

# **CHEMICAL INFORMATION** (if available)
# [Include ChEMBL ID, molecule type, and other chemical details if available]

# **Sources:** {sources}

# *This application is for clinical decision support and should only be used by qualified healthcare professionals.*

# ---

# REFERENCE TEXT TO USE:
# {context}

# AVAILABLE KNOWLEDGE BASE SOURCES:
# {sources}

# DRUG QUERY:
# {patient_data}

# PREVIOUS CONVERSATION:
# {chat_history}

# YOUR TASK:
# Provide comprehensive drug information based on the knowledge base. Focus only on factual information from the provided sources. If the requested drug is not found in the knowledge base, clearly state this and suggest verifying the drug name spelling.
# """

# GENERAL_PROMPT = """
# You are a medical AI assistant. Answer the user's question based ONLY on the provided reference materials from the knowledge base.

# IMPORTANT: You must follow these formatting rules:

# 1. **Opening Context**: Start with 1-2 sentences providing brief context about the medical topic being asked about.

# 2. **Main Content**: Organize your response using clear headings and bullet points when listing symptoms, treatments, or information:
#    - Use headings like "Common symptoms:", "Signs that suggest [condition]:", "Treatment options:", etc.
#    - Use bullet points for lists of symptoms, signs, or recommendations
#    - Be direct and factual - no conversational greetings or filler words

# 3. **References**: Include specific citations when stating medical facts, guidelines, or recommendations using the format [Source: document_name, page/section]

# 4. **Warnings**: If discussing serious conditions, end with appropriate medical warnings.

# 5. **Knowledge Base Only**: Base your entire response on the provided reference materials.

# ---

# REFERENCE TEXT TO USE:
# {context}

# AVAILABLE KNOWLEDGE BASE SOURCES:
# {sources}

# USER'S QUERY:
# {patient_data}

# PREVIOUS CONVERSATION:
# {chat_history}

# YOUR TASK:
# Answer the user's question following the formatting rules above. Provide factual, well-structured information based solely on the knowledge base sources.
# """

# def classify_query_type(query: str, chat_history: str = "") -> str:
#     """
#     Classify the type of query to determine which prompt template to use.
    
#     Args:
#         query: The user's current query
#         chat_history: Previous conversation context
        
#     Returns:
#         str: Query type ('differential_diagnosis', 'drug_information', or 'general')
#     """
#     query_lower = query.lower()
#     combined_text = f"{chat_history} {query}".lower()
    
#     # Drug information keywords
#     drug_keywords = [
#         'drug', 'medication', 'medicine', 'tablet', 'capsule', 'injection',
#         'side effects', 'adverse effects', 'contraindications', 'interactions',
#         'dosage', 'dose', 'prescribe', 'prescription', 'pharmaceutical',
#         'mg', 'ml', 'mcg', 'units', 'twice daily', 'once daily',
#         'paracetamol', 'ibuprofen', 'amoxicillin', 'metformin', 'aspirin',
#         'antibiotic', 'analgesic', 'anti-inflammatory', 'antacid'
#     ]
    
#     # Differential diagnosis keywords
#     diagnosis_keywords = [
#         'diagnose', 'diagnosis', 'differential', 'symptoms', 'signs',
#         'patient presents', 'chief complaint', 'history of present illness',
#         'physical examination', 'vital signs', 'fever', 'pain', 'cough',
#         'shortness of breath', 'chest pain', 'abdominal pain', 'headache',
#         'nausea', 'vomiting', 'diarrhea', 'rash', 'swelling',
#         'what condition', 'what disease', 'likely diagnosis', 'rule out',
#         'year old', 'presents with', 'complains of', 'reports'
#     ]
    
#     # Check for drug information queries
#     drug_score = sum(1 for keyword in drug_keywords if keyword in query_lower)
    
#     # Check for diagnosis queries
#     diagnosis_score = sum(1 for keyword in diagnosis_keywords if keyword in query_lower)
    
#     # Additional patterns for diagnosis
#     diagnosis_patterns = [
#         r'\d+\s*(year|yr)\s*old',  
#         r'patient.*with',  
#         r'presents.*with',  
#         r'complains.*of',  #
#         r'history.*of', 
#         r'vital signs?',  
#         r'bp\s*\d+',  
#         r'temp\w*\s*\d+',  
#         r'hr\s*\d+',  
#     ]
    
#     pattern_matches = sum(1 for pattern in diagnosis_patterns if re.search(pattern, query_lower))
    
#     # Decision logic
#     if drug_score >= 2 or any(word in query_lower for word in ['what is', 'tell me about'] + [kw for kw in drug_keywords[:10]]):
#         return 'drug_information'
#     elif diagnosis_score >= 2 or pattern_matches >= 1 or any(phrase in query_lower for phrase in ['patient', 'symptoms', 'diagnosis', 'presents with']):
#         return 'differential_diagnosis'
#     else:
#         return 'general'



# def get_prompt_template(query_type: str) -> str:
#     """
#     Get the appropriate prompt template based on query type.
    
#     Args:
#         query_type: Type of query ('differential_diagnosis', 'drug_information', or 'general')
        
#     Returns:
#         str: The appropriate prompt template
#     """
#     if query_type == 'differential_diagnosis':
#         return DIFFERENTIAL_DIAGNOSIS_PROMPT
#     elif query_type == 'drug_information':
#         return DRUG_INFORMATION_PROMPT
#     else:
#         return GENERAL_PROMPT

# def is_diagnosis_complete(response: str) -> bool:
#     """Check if diagnosis process is complete (no more questions needed)."""
#     response_lower = response.lower().strip()
#     # Check for various question indicators
#     question_indicators = [
#         "question:", 
#         "additional information needed",
#         "need more information",
#         "please provide",
#         "can you tell me",
#         "do you have"
#     ]
#     return not any(indicator in response_lower for indicator in question_indicators)

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# async def generate_response(query: str, chat_history: str, patient_data: str, retriever):
#     """Generate a response using the LLM with appropriate prompt template based on query type."""
#     try:
#         start_time = time.time()
        
#         # Classify the query type
#         query_type = classify_query_type(query, chat_history)
#         print(f"Query classified as: {query_type}")
        
#         # Get the appropriate prompt template
#         prompt_template = get_prompt_template(query_type)
        
#         # Retrieve context and actual sources
#         context, actual_sources = retrieve_context(query, patient_data, retriever)
        
#         # For differential diagnosis, count prior questions (legacy feature)
#         if query_type == 'differential_diagnosis':
#             question_count = chat_history.count("Question:") if chat_history else 0
#             if question_count >= 7:
#                 context += "\nNote: Maximum of 7 questions reached; provide an assessment with available data."
        
#         # Format sources for display
#         sources_text = ", ".join(actual_sources) if actual_sources else "No sources available"
        
#         # Populate prompt with the selected template
#         prompt = prompt_template.format(
#             patient_data=patient_data,
#             context=context,
#             sources=sources_text,
#             chat_history=chat_history or "No previous conversation",
#         )

#         # Generate model response with vertex AI
#         model = GenerativeModel("gemini-2.5-pro")
#         generation_config = GenerationConfig(
#             temperature=0.4, 
#             max_output_tokens=3000,  
#             top_p=0.95,
#         )
        
#         start = time.time()
#         response = await model.generate_content_async(prompt, generation_config=generation_config)
#         inference_time = time.time() - start
#         print(f"Inference time: {inference_time:.2f}s")
        
#         response_text = response.text
#         diagnosis_complete = is_diagnosis_complete(response_text)
        
#         return response_text, diagnosis_complete
    
#     except exceptions.ResourceExhausted:
#         raise HTTPException(status_code=429, detail="Rate limit exceeded or resource unavailable, please try again later.")
#     except exceptions.GoogleAPIError as e:
#         raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
#     except Exception as e:
#         print(f"Unexpected error in generate_response: {str(e)}")
#         import traceback
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

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

async def retrieve_context(query: str, patient_data: str, retriever: Any) -> str:
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
async def generate_response(query: str, chat_history: str, patient_data: str, retriever: Any):
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
        model = GenerativeModel("gemini-2.5-pro")
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
    
    