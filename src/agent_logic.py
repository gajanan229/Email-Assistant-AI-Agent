import streamlit as st # Only needed for the cache decorator
import os
import re
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from typing import List, Optional, Tuple

# Keep this import if FAISS is used, adjust if using a different vector store
from langchain.vectorstores.base import VectorStoreRetriever


# --- Core Functions ---

def load_api_key() -> Optional[str]:
    """Loads the OpenAI API key from the .env file.

    Returns:
        Optional[str]: The API key if found, otherwise None.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file.")
        return None
    print("INFO: OpenAI API Key loaded successfully.")
    return api_key

@st.cache_resource # Keep cache for expensive LLM object
def configure_llm(api_key: str) -> Optional[OpenAI]:
    """Configures and returns the Langchain LLM, cached for efficiency.

    Args:
        api_key (str): The OpenAI API key.

    Returns:
        Optional[OpenAI]: The configured LLM instance or None if configuration fails.
    """
    try:
        # Set a higher max_tokens limit to prevent abrupt output cutoff.
        llm = OpenAI(temperature=0.7, openai_api_key=api_key, max_tokens=1024)
        print("INFO: LLM configured successfully.")
        return llm
    except Exception as e:
        print(f"ERROR: Error configuring LLM: {e}")
        return None

def create_input_prompt_template() -> PromptTemplate:
    """Creates a basic prompt template for email generation without RAG.

    Returns:
        PromptTemplate: The Langchain prompt template instance.
    """
    template = """
    You are an AI assistant helping to draft a job application email.
    Generate a personalized email based on the following details:

    Position Applying For: {position}
    Company Name: {company}
    Desired Tone: {tone}
    Additional Notes/Instructions from User: {notes}

    **Instructions:**
    1.  Start with a professional salutation (e.g., "Dear Hiring Manager,").
    2.  **Craft an Engaging Opening:** Immediately after the salutation, write a compelling opening sentence or two. *Avoid generic phrases* like "I am writing to apply..." or "I am excited to apply...". Instead, try one of these approaches:
        *   Directly state your enthusiasm for this specific {position} at {company}, perhaps mentioning *why* it excites you based on the role title or company name itself (be creative!).
        *   If mentioned in the User Notes, lead with a *hook* related to a key skill or experience highly relevant to the {position}.
        *   Clearly state the role you're applying for within this engaging opening, but weave it in naturally.
    3.  **Connect to User Notes:** Weave in specific points, projects, or skills mentioned in the 'Additional Notes' into the body of the email, demonstrating their relevance to the {position}.
    4.  **Express Value:** Briefly explain *why* you believe you are a strong candidate, connecting your general profile (implied by the request) to the likely needs of the role.
    5.  **Maintain Tone:** Ensure the entire email reflects the specified {tone}.
    6.  **Closing:** End with a professional closing (e.g., "Sincerely,"), followed by "[Your Name]".

    Generated Email:
    """
    prompt = PromptTemplate(template=template, input_variables=["position", "company", "tone", "notes"])
    return prompt

# This might not need caching if only used as a fallback.
# @st.cache_resource
def create_input_llm_chain(_llm: OpenAI, _prompt_template: PromptTemplate) -> Optional[LLMChain]:
    """Creates a basic Langchain LLMChain.

    Args:
        _llm (OpenAI): The configured LLM instance.
        _prompt_template (PromptTemplate): The prompt template to use.

    Returns:
        Optional[LLMChain]: The created chain or None if inputs are invalid.
    """
    if _llm and _prompt_template:
        # Note: LLMChain is deprecated, consider replacing with `prompt | llm` RunnableSequence
        print("INFO: Creating basic LLM Chain (deprecated).")
        return LLMChain(llm=_llm, prompt=_prompt_template)
    print("ERROR: Could not create basic LLM Chain due to invalid inputs.")
    return None


# --- RAG Prompt Template & Parser ---

def create_rag_prompt_template_with_parser() -> Optional[Tuple[PromptTemplate, StructuredOutputParser]]:
    """Creates the RAG prompt template and a structured output parser.

    Defines the desired JSON output structure (subject, body) and includes
    instructions for the LLM within the prompt.

    Returns:
        Optional[Tuple[PromptTemplate, StructuredOutputParser]]: The prompt template and
            the configured output parser, or None on failure.
    """
    try:
        response_schemas = [
            ResponseSchema(name="subject", description="The suggested subject line for the job application email."),
            ResponseSchema(name="body", description="The full body text of the job application email, starting with the salutation and ending with the closing and placeholder for the sender's name.")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        template = """
        You are an expert AI assistant specializing in crafting **compelling, narrative-style** job application emails (Subject and Body). Your goal is to connect with the reader on a deeper level, beyond just listing qualifications.
        Your tone should be: {tone}.
        The user wants to apply for the position of: {position} at {company}.
        User's specific notes/instructions: {notes}

        Use the following relevant context extracted from the user's RESUME:
        --- BEGIN RESUME CONTEXT ---
        {context_resume}
        --- END RESUME CONTEXT ---

        Use the following relevant context extracted from the JOB DESCRIPTION:
        --- BEGIN JOB DESCRIPTION CONTEXT ---
        {context_jd}
        --- END JOB DESCRIPTION CONTEXT ---

        **Instructions for Generating the Email Body:**

        1.  **Craft an Opening Narrative (1-2 Paragraphs):**
            *   **Identify the Core Theme:** Analyze the **Job Description Context** to identify the company's core mission, a significant challenge they address, a key innovation they are driving, or the broader impact of the work.
            *   **Establish Personal Resonance:** Begin the email body (immediately after the salutation) by reflecting on this core theme. Explain *why* this mission, challenge, or impact resonates deeply with *your* (the user's implied) values, passion, philosophy, or long-term goals. Draw inspiration from the **Resume Context** or **User Notes** if they support this connection. Make it sound genuine and specific to **{company}**.
            *   **Connect Skills Thematically:** Briefly mention 1-2 high-level skill areas or experiences (from **Resume Context**) not as a list, but as tools or passions relevant to contributing to this core theme. Example: "My passion for [Skill Area from Resume] aligns perfectly with [Company]'s drive to solve [Problem from JD Context]."
            *   **CRITICAL:** **DO NOT explicitly state "I am applying for the {position}" or use phrases like "I am writing to apply..." or "I was excited to see the opening..." in this initial narrative section.** The purpose is to establish connection and interest first.

        2.  **Transition and Introduce the Role:**
            *   After establishing the thematic connection, create a smooth transition.
            *   *Then*, naturally introduce the specific **{position}** role as the concrete opportunity you are pursuing to *actively contribute* to the mission/challenge you just discussed. Frame it as the logical next step given your resonance with the company's goals. Example: "It's with this shared vision in mind that I am particularly drawn to the **{position}** opportunity at **{company}**."

        3.  **Provide Supporting Evidence:**
            *   Select 1-2 specific examples, projects, or quantifiable achievements from the **Resume Context** that directly support your ability to succeed in the **{position}** and contribute to the goals mentioned in the **Job Description Context**. Link them clearly to the requirements.

        4.  **Incorporate User Notes:**
            *   Thoughtfully integrate any specific points from the user's '{notes}' where they best enhance the narrative or provide crucial details.

        5.  **Maintain Tone and Professionalism:**
            *   Ensure the entire email body consistently reflects the specified {tone} while remaining professional.

        6.  **Closing:**
            *   Conclude with a forward-looking statement expressing strong enthusiasm for the opportunity and the next steps in the process. Reiterate your interest in contributing to **{company}**.
            *   Use a professional closing (e.g., "Sincerely,") followed by the placeholder "[Your Name]".

        **Instructions for Generating the Subject Line:**
        *   Create a concise and compelling subject line. Include the **{position}** title and **{company}**. Consider adding your name or a *very brief* highlight. Examples: "Application: {position} - {company}", "Enthusiastic Application for {position} at {company} - [Your Name]", "{position} at {company} - Interest in [Key Area from JD/Notes]".

        **Output Format:**
        Provide the output ONLY in the following JSON format, ensuring the 'body' contains only the text after the salutation and before the final closing/name:
        {format_instructions}
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["tone", "position", "company", "notes", "context_resume", "context_jd"],
            partial_variables={"format_instructions": format_instructions}
        )
        print("INFO: RAG prompt template and parser created successfully.")
        return prompt, output_parser
    except Exception as e:
        print(f"ERROR: Failed to create RAG prompt template or parser: {e}")
        return None

# --- Helper Function for Formatting Retrieved Docs ---

def format_retrieved_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single string for context.

    Args:
        docs (List[Document]): List of documents retrieved from the vector store.

    Returns:
        str: A single string containing the content of the documents, separated,
             or "N/A" if no documents were provided.
    """
    if not docs:
        return "N/A"
    # Join page content with separators, stripping extra whitespace.
    return "\n---\n".join([doc.page_content.strip() for doc in docs])


# --- Key Skill Extraction Function ---

def extract_key_skills(jd_full_text: str, llm: OpenAI) -> Optional[str]:
    """Extracts key skills from the job description text using the LLM.

    Args:
        jd_full_text (str): The full text content of the job description.
        llm (OpenAI): The configured LLM instance.

    Returns:
        Optional[str]: A string containing the extracted skills (likely a list),
             or None if extraction fails or inputs are invalid.
    """
    if not jd_full_text:
        print("WARNING: Job description text is empty, cannot extract skills.")
        return None
    if not llm:
        print("ERROR: LLM not available, cannot extract skills.")
        return None

    skill_prompt_text = f"""
    Based on the following job description text, please list the top 5-7 most important skills, qualifications, or requirements mentioned.
    Present them as a concise bulleted or numbered list.

    Job Description Text:
    ---
    {jd_full_text}
    ---

    Key Skills/Requirements:
    """
    try:
        print("DEBUG: Calling LLM for skill extraction...")
        response = llm.invoke(skill_prompt_text)
        print("DEBUG: Skill extraction response received.")
        return response.strip() if response else "Could not extract skills."
    except Exception as e:
        print(f"ERROR: Error extracting skills via LLM: {e}")
        return None

# --- Output Parsing Functions ---

def parse_subject_body_fallback(response: str) -> Tuple[str, str]:
    """Fallback parser using regex to find Subject: and Body: if structured parsing fails.

    Args:
        response (str): The raw string response from the LLM.

    Returns:
        Tuple[str, str]: The extracted subject and body strings. Defaults will be
                         returned if parsing is unsuccessful.
    """
    print("DEBUG: Using fallback parser.")
    subject = "Subject Not Found (Fallback)"
    body = response # Default to full response if parsing fails

    # Try to find "Subject:" case-insensitively, allowing multi-line content.
    subject_match = re.search(r"Subject:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
    if subject_match:
        subject_content = subject_match.group(1).strip()
        # Try to find "Body:" after the subject line/block.
        body_match = re.search(r"Body:\s*(.*)", subject_content, re.IGNORECASE | re.DOTALL)
        if body_match:
            # Subject is the part before "Body:", body is the part after.
            subject = subject_content.split("Body:")[0].strip()
            body = body_match.group(1).strip()
        else:
            # Assume Subject is the first line and the rest is body if "Body:"
            # marker is missing after "Subject:".
            lines = subject_content.split('\n', 1)
            subject = lines[0]
            if len(lines) > 1:
                body = lines[1].strip()
            else:
                body = "Body Not Found (Fallback)"

    else:
        # If "Subject:" wasn't found, try finding "Body:" directly.
        body_match = re.search(r"Body:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
        if body_match:
            body = body_match.group(1).strip()
            # Assume subject might be the text before "Body:", or missing.
            subject = response.split("Body:")[0].strip()
            if not subject or subject.lower().startswith("body:"):
                 subject = "Subject Not Found (Fallback)"

    # Clean up potential markdown code block fences or JSON artifacts.
    subject = re.sub(r"```(json)?", "", subject).strip()
    body = re.sub(r"```(json)?", "", body).strip()

    # Remove potential leading/trailing quotes if parser failed.
    if subject.startswith('"') and subject.endswith('"'): subject = subject[1:-1]
    if body.startswith('"') and body.endswith('"'): body = body[1:-1]

    print(f"DEBUG Fallback - Subject: {subject[:50]}...")
    print(f"DEBUG Fallback - Body: {body[:100]}...")
    return subject, body


def parse_llm_output(response: str, parser: StructuredOutputParser) -> Tuple[str, str]:
   """Parses the LLM output using the structured parser with a regex fallback.

   Args:
        response (str): The raw string response from the LLM.
        parser (StructuredOutputParser): The Langchain parser instance.

   Returns:
        Tuple[str, str]: The extracted subject and body strings.
   """
   try:
       print("DEBUG: Attempting structured parsing...")
       parsed_output = parser.parse(response)
       subject = parsed_output.get('subject', 'Subject Not Found (Parsed)')
       body = parsed_output.get('body', 'Body Not Found (Parsed)')
       print("DEBUG: Structured parsing successful.")
       return subject, body
   except Exception as e:
       print(f"WARNING: Structured parsing failed: {e}. Using fallback regex parser.")
       return parse_subject_body_fallback(response)


# --- RAG Generation Function ---

def generate_rag_email(llm: OpenAI, rag_prompt_template: PromptTemplate, output_parser: StructuredOutputParser, position: str, company: str, tone: str, notes: str, resume_retriever: VectorStoreRetriever, jd_retriever: VectorStoreRetriever) -> Tuple[Optional[str], Optional[str]]:
    """Generates email using RAG: retrieves context, formats prompt, calls LLM, parses output.

    Args:
        llm (OpenAI): The configured LLM instance.
        rag_prompt_template (PromptTemplate): The RAG-specific prompt template.
        output_parser (StructuredOutputParser): The parser for the LLM response.
        position (str): The job position title.
        company (str): The company name.
        tone (str): The desired email tone.
        notes (str): Additional user notes.
        resume_retriever (VectorStoreRetriever): The retriever for the resume vector store.
        jd_retriever (VectorStoreRetriever): The retriever for the job description vector store.

    Returns:
        Tuple[Optional[str], Optional[str]]: The generated subject and body strings,
                                             or (None, None) on failure.
    """
    print("INFO: Starting RAG generation...")

    if not resume_retriever or not jd_retriever:
        print("ERROR: Cannot generate RAG email without both resume and JD retrievers.")
        return None, None

    # 1. Construct Query for Retrievers
    query = f"Relevant skills, experience, and qualifications for the {position} role at {company}. Specific points to consider: {notes if notes else 'None'}"
    print(f"DEBUG: RAG Query: {query}")

    # 2. Retrieve Relevant Docs using .invoke()
    try:
        print("DEBUG: Retrieving resume context...")
        resume_docs = resume_retriever.invoke(query)
        print(f"DEBUG: Retrieved {len(resume_docs)} resume snippets.")
    except Exception as e:
        print(f"ERROR: Error retrieving resume context: {e}")
        resume_docs = []

    try:
        print("DEBUG: Retrieving job description context...")
        jd_docs = jd_retriever.invoke(query)
        print(f"DEBUG: Retrieved {len(jd_docs)} JD snippets.")
    except Exception as e:
        print(f"ERROR: Error retrieving job description context: {e}")
        jd_docs = []

    # 3. Format Contexts for Prompt
    resume_context = format_retrieved_docs(resume_docs)
    jd_context = format_retrieved_docs(jd_docs)

    # 4. Prepare Input Data for RAG Prompt
    input_data = {
        "position": position,
        "company": company,
        "tone": tone,
        "notes": notes if notes else "None",
        "context_resume": resume_context,
        "context_jd": jd_context
        # format_instructions are handled by partial_variables in the template
    }

    # 5. Generate Email using LLM and RAG Prompt
    try:
        # Format the prompt string with all context and inputs.
        final_prompt_string = rag_prompt_template.format(**input_data)
        print("DEBUG: Sending final prompt to LLM...")
        # Use .invoke() for the LLM call.
        raw_response = llm.invoke(final_prompt_string)
        print("DEBUG: Received raw response from LLM.")
        # print(f"DEBUG Raw Response:\n{raw_response}\n") # Optional: Log raw response

        # 6. Parse Output
        subject, body = parse_llm_output(raw_response, output_parser)
        print("INFO: RAG email generation and parsing complete.")
        return subject, body

    except Exception as e:
        print(f"ERROR: Error during RAG email synthesis or parsing: {e}")
        if "api key" in str(e).lower():
            print("ERROR HINT: Check your OpenAI API key validity and credits.")
        return None, None # Indicate failure