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
from typing import Dict, Tuple
from google.api_core import exceptions
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Prompt templates for different query types
DIFFERENTIAL_DIAGNOSIS_PROMPT = """
You are an expert medical AI, designed to assist a qualified doctor. Your task is to analyze patient information and provide a structured clinical assessment based ONLY on the provided reference materials.

IMPORTANT: You must follow all rules listed below.

---
RULES:
1.  **Source of Truth**: Base your entire analysis strictly on the `REFERENCE TEXT TO USE`. Do NOT reference sources that are not explicitly provided in the knowledge base.
2.  **Selective Citations**: Only cite sources when:
    - Making specific clinical recommendations or diagnostic criteria
    - Stating clinical guidelines or protocols
    - Referencing specific diagnostic tests or treatment approaches
    - Quoting or paraphrasing specific clinical information
   Do NOT cite sources for general medical knowledge, basic pathophysiology, or common clinical observations.
3.  **Accurate Citations**: When citing, use ONLY the exact source names from the `AVAILABLE KNOWLEDGE BASE SOURCES` list. Include page numbers, section names, or specific document identifiers when available in the format: `[Source: document_name, page/section]`.
4.  **Medical Acronyms**: Interpret and correctly use common medical acronyms.
5.  **Critical Assessment**: Always begin by checking for life-threatening conditions. If any vital signs are in a dangerous range, issue an "CRITICAL ALERT" immediately and prioritize urgent interventions.
6.  **Output Format**: Structure your response using the following format:

**CLINICAL OVERVIEW**
[Provide a brief 1-2 paragraph overview summarizing the case in relation to the question or user query, highlighting key clinical features and initial impression]

**MOST LIKELY DIAGNOSIS**
[Present the primary diagnosis with brief supporting rationale]

**DIFFERENTIAL DIAGNOSES**
1. **[Primary Diagnosis]**: [Brief explanation with supporting/opposing evidence, cite specific guidelines when applicable]
2. **[Secondary Diagnosis]**: [Brief explanation with supporting/opposing evidence, cite specific guidelines when applicable]  
3. **[Tertiary Diagnosis]**: [Brief explanation with supporting/opposing evidence, cite specific guidelines when applicable]
[Add more if clinically relevant - aim for 3+ when possible]

**⚡ IMMEDIATE WORKUP & INVESTIGATIONS**
- [List essential immediate tests/investigations needed]
- [Include time-sensitive protocols if applicable]
- [Prioritize based on clinical urgency]

**MANAGEMENT & RECOMMENDATIONS**
- [Immediate management steps]
- [Treatment recommendations with specific interventions]
- [Monitoring requirements]
- [Follow-up plans]

**RED FLAGS / DANGER SIGNS**
- [List warning signs that require immediate attention]
- [Critical deterioration indicators]

**ADDITIONAL INFORMATION NEEDED** (if applicable)
[Only include this section if you need more critical information to refine the diagnosis or management plan. Ask one focused, essential question.]

**Sources:** {sources}

*This application is for clinical decision support and should only be used by qualified healthcare professionals.*

---

REFERENCE TEXT TO USE:
{context}

AVAILABLE KNOWLEDGE BASE SOURCES:
{sources}

PATIENT'S CURRENT INFORMATION:
{patient_data}

PREVIOUS CONVERSATION:
{chat_history}

YOUR TASK:
Evaluate the patient's information according to the rules and provide your response in the specified format. Use emojis as section headers as shown. Focus on practical, actionable clinical guidance. Remember: cite only when referencing specific clinical guidelines, protocols, or criteria from the provided sources. Do not cite general medical knowledge or make up source names.
"""

DRUG_INFORMATION_PROMPT = """
You are an expert pharmacology AI, designed to assist qualified healthcare professionals with drug information. Your task is to provide comprehensive drug information based ONLY on the provided reference materials from the drug knowledge base.

IMPORTANT: You must follow all rules listed below.

---
RULES:
1.  **Source of Truth**: Base your entire response strictly on the `REFERENCE TEXT TO USE`. Do NOT make up or infer information not explicitly provided in the knowledge base.
2.  **Accurate Information**: Only provide information that is directly found in the drug database. If specific information is not available, clearly state "Information not available in knowledge base."
3.  **Accurate Citations**: When citing, use ONLY the exact source names from the `AVAILABLE KNOWLEDGE BASE SOURCES` list. For drug database sources, include the full citation with URL when available.
4.  **Drug Names**: Use the exact drug names as they appear in the knowledge base.
5.  **Output Format**: Structure your response using the following format:

**DRUG OVERVIEW**
[Provide a brief overview of the drug based on the knowledge base, including what it is, active ingredients, and primary indications]

**SIDE EFFECTS**
[List side effects by frequency categories as found in the knowledge base:
- Very Common (>1/10)
- Common (1/100 to 1/10) 
- Uncommon (1/1,000 to 1/100)
- Rare (1/10,000 to 1/1,000)
- Very Rare (<1/10,000)
- Unknown frequency
Only include categories that have documented side effects in the knowledge base]

**DRUG INTERACTIONS**
[List known drug interactions from the knowledge base. If none are documented, state "No drug interactions documented in knowledge base."]

**CONTRAINDICATIONS**
[List contraindications from the knowledge base. If none are documented, state "No contraindications documented in knowledge base."]

**MECHANISM OF ACTION** (if available)
[Include molecular targets, mechanisms of action, and pharmacological information if available in the knowledge base]

**CHEMICAL INFORMATION** (if available)
[Include ChEMBL ID, molecule type, and other chemical details if available]

**Sources:** {sources}

*This application is for clinical decision support and should only be used by qualified healthcare professionals.*

---

REFERENCE TEXT TO USE:
{context}

AVAILABLE KNOWLEDGE BASE SOURCES:
{sources}

DRUG QUERY:
{patient_data}

PREVIOUS CONVERSATION:
{chat_history}

YOUR TASK:
Provide comprehensive drug information based on the knowledge base. Focus only on factual information from the provided sources. If the requested drug is not found in the knowledge base, clearly state this and suggest verifying the drug name spelling.
"""

GENERAL_PROMPT = """
You are a medical AI assistant. Answer the user's question based ONLY on the provided reference materials from the knowledge base.

IMPORTANT: You must follow these formatting rules:

1. **Opening Context**: Start with 1-2 sentences providing brief context about the medical topic being asked about.

2. **Main Content**: Organize your response using clear headings and bullet points when listing symptoms, treatments, or information:
   - Use headings like "Common symptoms:", "Signs that suggest [condition]:", "Treatment options:", etc.
   - Use bullet points for lists of symptoms, signs, or recommendations
   - Be direct and factual - no conversational greetings or filler words

3. **References**: Include specific citations when stating medical facts, guidelines, or recommendations using the format [Source: document_name, page/section]

4. **Warnings**: If discussing serious conditions, end with appropriate medical warnings.

5. **Knowledge Base Only**: Base your entire response on the provided reference materials.

---

REFERENCE TEXT TO USE:
{context}

AVAILABLE KNOWLEDGE BASE SOURCES:
{sources}

USER'S QUERY:
{patient_data}

PREVIOUS CONVERSATION:
{chat_history}

YOUR TASK:
Answer the user's question following the formatting rules above. Provide factual, well-structured information based solely on the knowledge base sources.
"""

def classify_query_type(query: str, chat_history: str = "") -> str:
    """
    Classify the type of query to determine which prompt template to use.
    
    Args:
        query: The user's current query
        chat_history: Previous conversation context
        
    Returns:
        str: Query type ('differential_diagnosis', 'drug_information', or 'general')
    """
    query_lower = query.lower()
    combined_text = f"{chat_history} {query}".lower()
    
    # Drug information keywords
    drug_keywords = [
        'drug', 'medication', 'medicine', 'tablet', 'capsule', 'injection',
        'side effects', 'adverse effects', 'contraindications', 'interactions',
        'dosage', 'dose', 'prescribe', 'prescription', 'pharmaceutical',
        'mg', 'ml', 'mcg', 'units', 'twice daily', 'once daily',
        'paracetamol', 'ibuprofen', 'amoxicillin', 'metformin', 'aspirin',
        'antibiotic', 'analgesic', 'anti-inflammatory', 'antacid'
    ]
    
    # Differential diagnosis keywords
    diagnosis_keywords = [
        'diagnose', 'diagnosis', 'differential', 'symptoms', 'signs',
        'patient presents', 'chief complaint', 'history of present illness',
        'physical examination', 'vital signs', 'fever', 'pain', 'cough',
        'shortness of breath', 'chest pain', 'abdominal pain', 'headache',
        'nausea', 'vomiting', 'diarrhea', 'rash', 'swelling',
        'what condition', 'what disease', 'likely diagnosis', 'rule out',
        'year old', 'presents with', 'complains of', 'reports'
    ]
    
    # Check for drug information queries
    drug_score = sum(1 for keyword in drug_keywords if keyword in query_lower)
    
    # Check for diagnosis queries
    diagnosis_score = sum(1 for keyword in diagnosis_keywords if keyword in query_lower)
    
    # Additional patterns for diagnosis
    diagnosis_patterns = [
        r'\d+\s*(year|yr)\s*old',  
        r'patient.*with',  
        r'presents.*with',  
        r'complains.*of',  #
        r'history.*of', 
        r'vital signs?',  
        r'bp\s*\d+',  
        r'temp\w*\s*\d+',  
        r'hr\s*\d+',  
    ]
    
    pattern_matches = sum(1 for pattern in diagnosis_patterns if re.search(pattern, query_lower))
    
    # Decision logic
    if drug_score >= 2 or any(word in query_lower for word in ['what is', 'tell me about'] + [kw for kw in drug_keywords[:10]]):
        return 'drug_information'
    elif diagnosis_score >= 2 or pattern_matches >= 1 or any(phrase in query_lower for phrase in ['patient', 'symptoms', 'diagnosis', 'presents with']):
        return 'differential_diagnosis'
    else:
        return 'general'



def get_prompt_template(query_type: str) -> str:
    """
    Get the appropriate prompt template based on query type.
    
    Args:
        query_type: Type of query ('differential_diagnosis', 'drug_information', or 'general')
        
    Returns:
        str: The appropriate prompt template
    """
    if query_type == 'differential_diagnosis':
        return DIFFERENTIAL_DIAGNOSIS_PROMPT
    elif query_type == 'drug_information':
        return DRUG_INFORMATION_PROMPT
    else:
        return GENERAL_PROMPT

def is_diagnosis_complete(response: str) -> bool:
    """Check if diagnosis process is complete (no more questions needed)."""
    response_lower = response.lower().strip()
    # Check for various question indicators
    question_indicators = [
        "question:", 
        "additional information needed",
        "need more information",
        "please provide",
        "can you tell me",
        "do you have"
    ]
    return not any(indicator in response_lower for indicator in question_indicators)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_response(query: str, chat_history: str, patient_data: str, retriever):
    """Generate a response using the LLM with appropriate prompt template based on query type."""
    try:
        start_time = time.time()
        
        # Classify the query type
        query_type = classify_query_type(query, chat_history)
        print(f"Query classified as: {query_type}")
        
        # Get the appropriate prompt template
        prompt_template = get_prompt_template(query_type)
        
        # Retrieve context and actual sources
        context, actual_sources = retrieve_context(query, patient_data, retriever)
        
        # For differential diagnosis, count prior questions (legacy feature)
        if query_type == 'differential_diagnosis':
            question_count = chat_history.count("Question:") if chat_history else 0
            if question_count >= 7:
                context += "\nNote: Maximum of 7 questions reached; provide an assessment with available data."
        
        # Format sources for display
        sources_text = ", ".join(actual_sources) if actual_sources else "No sources available"
        
        # Populate prompt with the selected template
        prompt = prompt_template.format(
            patient_data=patient_data,
            context=context,
            sources=sources_text,
            chat_history=chat_history or "No previous conversation",
        )

        # Generate model response with vertex AI
        model = GenerativeModel("gemini-2.5-pro")
        generation_config = GenerationConfig(
            temperature=0.4, 
            max_output_tokens=4000,  
            top_p=0.95,
        )
        
        start = time.time()
        response = await model.generate_content_async(prompt, generation_config=generation_config)
        inference_time = time.time() - start
        print(f"Inference time: {inference_time:.2f}s")
        
        response_text = response.text
        diagnosis_complete = is_diagnosis_complete(response_text)
        
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


# Test function for query classification (for development/debugging)
def test_query_classification():
    """Test function to demonstrate query classification."""
    test_queries = [
        "Patient presents with fever and cough for 3 days",
        "What are the side effects of amoxicillin?",
        "Tell me about paracetamol dosage",
        "What is tuberculosis?",
        "A 45-year-old male with chest pain",
        "Drug interactions with metformin",
        "What causes pneumonia?"
    ]
    
    for query in test_queries:
        query_type = classify_query_type(query)
        print(f"Query: '{query}' -> Type: {query_type}")

if __name__ == "__main__":
    test_query_classification()
    