import streamlit as st # Only needed for the cache decorator
import os
import re
from dotenv import load_dotenv
# Removed: from langchain.chains import LLMChain
from langchain.prompts import (
    PromptTemplate, # Still used for basic prompt if kept
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
# Replaced OpenAI with ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain.schema.messages import AIMessage # Import AIMessage for response handling
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
def configure_llm(api_key: str) -> Optional[ChatOpenAI]:
    """Configures and returns the Langchain Chat LLM, cached for efficiency.

    Args:
        api_key (str): The OpenAI API key.

    Returns:
        Optional[ChatOpenAI]: The configured ChatOpenAI instance or None if configuration fails.
    """
    try:
        # Use ChatOpenAI with the specified model
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=api_key,
            max_tokens=1024 # Keep max_tokens, useful for controlling output length budget
        )
        print("INFO: ChatOpenAI LLM (gpt-4o) configured successfully.")
        return llm
    except Exception as e:
        print(f"ERROR: Error configuring ChatOpenAI LLM: {e}")
        return None

# Note: create_input_prompt_template and create_input_llm_chain are less relevant
# now that the primary path uses ChatOpenAI and RAG. They are kept here
# but would need similar refactoring if used actively.

def create_input_prompt_template() -> PromptTemplate:
    """Creates a basic prompt template for email generation without RAG.
       (Legacy function, may need refactoring to ChatPromptTemplate if used).

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
    1.  Start with a professional salutation.
    2.  Clearly state the position being applied for and where it was seen (if mentioned in notes).
    3.  Briefly express enthusiasm for the role and company.
    4.  Mention any specific points from the 'Additional Notes'.
    5.  End with a professional closing and placeholder for the user's name.
    6.  Maintain the specified {tone}.

    Generated Email:
    """
    prompt = PromptTemplate(template=template, input_variables=["position", "company", "tone", "notes"])
    return prompt

# @st.cache_resource # Caching might not be needed if not primary path
def create_input_llm_chain(_llm: ChatOpenAI, _prompt_template: PromptTemplate) -> Optional[any]:
    """Creates a basic Langchain chain (needs update for Chat models).
       (Legacy function, LLMChain is deprecated, consider LCEL: prompt | llm).

    Args:
        _llm (ChatOpenAI): The configured LLM instance.
        _prompt_template (PromptTemplate): The prompt template to use.

    Returns:
        Optional[any]: The created chain or None if inputs are invalid.
                       Return type 'any' as LLMChain is deprecated.
    """
    if _llm and _prompt_template:
        print("INFO: Creating basic LLM Chain (deprecated, consider LCEL).")
        # from langchain.chains import LLMChain # Import locally if needed
        # return LLMChain(llm=_llm, prompt=_prompt_template)
        # Using LCEL (LangChain Expression Language) is preferred:
        # return _prompt_template | _llm
        print("WARNING: LLMChain is deprecated. Returning None for basic chain.")
        return None # Avoid using deprecated LLMChain
    print("ERROR: Could not create basic LLM Chain due to invalid inputs.")
    return None


# --- RAG Prompt Template & Parser ---

def create_rag_prompt_template_with_parser() -> Optional[Tuple[ChatPromptTemplate, StructuredOutputParser]]:
    """Creates the RAG ChatPromptTemplate and a structured output parser.

    Defines the desired JSON output structure (subject, body) and includes
    instructions for the LLM within the prompt using System and Human messages.

    Returns:
        Optional[Tuple[ChatPromptTemplate, StructuredOutputParser]]: The chat prompt template and
            the configured output parser, or None on failure.
    """
    try:
        response_schemas = [
            ResponseSchema(name="subject", description="The suggested subject line for the job application email."),
            ResponseSchema(name="body", description="The full body text of the job application email, starting with the salutation and ending with the closing and placeholder for the sender's name.")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        # System Message: Define the AI's role and overall goal.
        system_template = """You are an expert AI assistant specializing in crafting compelling, narrative-style job application emails. Your goal is to help the user create a personalized email based on their resume, the job description, and specific instructions."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        # Human Message: Provide context, specific instructions, and desired output format.
        human_template = """Please draft a job application email with the following details:
        - Tone: {tone}
        - Position Applying For: {position}
        - Company Name: {company}
        - User's Additional Notes: {notes}

        Use the following context from the user's RESUME:
        --- BEGIN RESUME CONTEXT ---
        {context_resume}
        --- END RESUME CONTEXT ---

        Use the following context from the JOB DESCRIPTION:
        --- BEGIN JOB DESCRIPTION CONTEXT ---
        {context_jd}
        --- END JOB DESCRIPTION CONTEXT ---

        **Instructions:**
        1. Based *only* on the provided context (Resume and Job Description) and the user's inputs (position, company, tone, notes), draft a compelling email Subject and Body.
        2. Highlight the alignment between the user's skills/experience (from Resume Context) and the key requirements/responsibilities (from Job Description Context). Be specific where possible.
        3. Naturally incorporate any relevant points from the user's 'notes'.
        4. Maintain the specified {tone} throughout the email body.
        5. The body should start with a professional salutation (e.g., "Dear Hiring Team," or specific name if provided in notes) and conclude with a professional closing statement and "[Your Name]".

        **Output Format:**
        Provide the output ONLY in the following JSON format:
        {format_instructions}
        """
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # Create the ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        # Update input variables (excluding partials)
        chat_prompt.input_variables = ["tone", "position", "company", "notes", "context_resume", "context_jd"]
        # Add partial variables
        chat_prompt.partial_variables = {"format_instructions": format_instructions}


        print("INFO: RAG ChatPromptTemplate and parser created successfully.")
        return chat_prompt, output_parser
    except Exception as e:
        print(f"ERROR: Failed to create RAG ChatPromptTemplate or parser: {e}")
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

def extract_key_skills(jd_full_text: str, llm: ChatOpenAI) -> Optional[str]:
    """Extracts key skills from the job description text using the Chat LLM.

    Args:
        jd_full_text (str): The full text content of the job description.
        llm (ChatOpenAI): The configured ChatOpenAI instance.

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

    # Construct the prompt text as before
    skill_prompt_text = f"""
    Based on the following job description text, please list the top 5-7 most important skills, qualifications, or requirements mentioned.
    Present them as a concise bulleted or numbered list.

    Job Description Text:
    ---
    {jd_full_text}
    ---

    Key Skills/Requirements:
    """
    # Wrap the prompt text in a HumanMessage for the Chat model
    skill_messages = [HumanMessage(content=skill_prompt_text)]

    try:
        print("DEBUG: Calling Chat LLM for skill extraction...")
        # Invoke the Chat LLM with the message list
        response_message = llm.invoke(skill_messages)

        # Extract content from the AIMessage response
        if isinstance(response_message, AIMessage):
            response_content = response_message.content
            print("DEBUG: Skill extraction response received.")
            return response_content.strip() if response_content else "Could not extract skills."
        else:
            print(f"ERROR: Unexpected response type from LLM during skill extraction: {type(response_message)}")
            return None

    except Exception as e:
        print(f"ERROR: Error extracting skills via Chat LLM: {e}")
        return None

# --- Output Parsing Functions ---

def parse_subject_body_fallback(response: str) -> Tuple[str, str]:
    """Fallback parser using regex to find Subject: and Body: if structured parsing fails.

    Args:
        response (str): The raw string response content from the LLM.

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


def parse_llm_output(response_content: str, parser: StructuredOutputParser) -> Tuple[str, str]:
   """Parses the LLM string output content using the structured parser with a regex fallback.

   Args:
        response_content (str): The raw string content from the LLM response.
        parser (StructuredOutputParser): The Langchain parser instance.

   Returns:
        Tuple[str, str]: The extracted subject and body strings.
   """
   try:
       print("DEBUG: Attempting structured parsing...")
       # Parse the string content
       parsed_output = parser.parse(response_content)
       subject = parsed_output.get('subject', 'Subject Not Found (Parsed)')
       body = parsed_output.get('body', 'Body Not Found (Parsed)')
       print("DEBUG: Structured parsing successful.")
       return subject, body
   except Exception as e:
       print(f"WARNING: Structured parsing failed: {e}. Using fallback regex parser.")
       # Pass the original string content to the fallback
       return parse_subject_body_fallback(response_content)


# --- RAG Generation Function ---

def generate_rag_email(llm: ChatOpenAI, rag_prompt_template: ChatPromptTemplate, output_parser: StructuredOutputParser, position: str, company: str, tone: str, notes: str, resume_retriever: VectorStoreRetriever, jd_retriever: VectorStoreRetriever) -> Tuple[Optional[str], Optional[str]]:
    """Generates email using RAG with Chat LLM: retrieves context, formats prompt, calls LLM, parses output.

    Args:
        llm (ChatOpenAI): The configured ChatOpenAI instance.
        rag_prompt_template (ChatPromptTemplate): The RAG-specific chat prompt template.
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
    print("INFO: Starting RAG generation with Chat LLM...")

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

    # 5. Generate Email using Chat LLM and RAG Prompt
    try:
        # Format the ChatPromptTemplate into a list of messages
        formatted_messages = rag_prompt_template.format_messages(**input_data)
        print("DEBUG: Sending formatted messages to Chat LLM...")
        # Invoke the Chat LLM with the formatted messages
        raw_response_message = llm.invoke(formatted_messages)
        print("DEBUG: Received raw response message from Chat LLM.")

        # Extract the string content from the AIMessage
        if isinstance(raw_response_message, AIMessage):
            raw_response_content = raw_response_message.content
            # print(f"DEBUG Raw Response Content:\n{raw_response_content}\n") # Optional log

            # 6. Parse Output content string
            subject, body = parse_llm_output(raw_response_content, output_parser)
            print("INFO: RAG email generation and parsing complete.")
            return subject, body
        else:
             print(f"ERROR: Unexpected response type from LLM: {type(raw_response_message)}")
             return None, None

    except Exception as e:
        print(f"ERROR: Error during RAG email synthesis or parsing with Chat LLM: {e}")
        if "api key" in str(e).lower():
            print("ERROR HINT: Check your OpenAI API key validity and credits.")
        return None, None # Indicate failure