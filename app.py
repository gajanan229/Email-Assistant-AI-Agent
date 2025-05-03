import streamlit as st
import pyperclip
# Import functions from the refactored modules
from src.doc_processor import (
    load_and_parse_document,
    chunk_documents,
    get_embeddings_model,
    create_vector_store_retriever
)
from src.agent_logic import (
    load_api_key,
    configure_llm,
    create_rag_prompt_template_with_parser,
    extract_key_skills,
    generate_rag_email
    # Removed create_input_prompt_template, create_input_llm_chain as fallback isn't primary
)

# --- Streamlit UI Configuration ---
st.set_page_config(layout="wide")
st.title("AI Job Application Email Assistant")

# --- Session State Initialization ---
# Initialize all required keys at the start if they don't exist.
default_values = {
    'setup_complete': False, # Combined flag for API key, LLM, Embeddings, Prompt/Parser
    'llm': None,
    'embeddings_model': None,
    'rag_prompt_template': None,
    'output_parser': None,
    'resume_docs': None, # Store raw docs if needed for other features
    'jd_docs': None,     # Store raw docs if needed for other features
    'resume_processed': False,
    'jd_processed': False,
    'resume_retriever': None,
    'jd_retriever': None,
    'current_resume_filename': None,
    'current_jd_filename': None,
    'key_skills': None,
    'generated_subject': "", # For displaying generated subject.
    'generated_body': ""     # For displaying generated body.
}
for key, default_value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Setup Phase: API Key, LLM, Embeddings, Prompt ---
# This block runs only once or until setup is complete.
if not st.session_state.setup_complete:
    st.info("Performing initial setup...")
    api_key = load_api_key()
    if not api_key:
        st.error("Setup failed: Could not load OpenAI API key. Check .env file.")
    else:
        llm = configure_llm(api_key)
        if not llm:
            st.error("Setup failed: Could not configure LLM. Check API key validity and credits.")
        else:
            st.session_state.llm = llm
            embeddings_model = get_embeddings_model()
            if not embeddings_model:
                st.error("Setup failed: Could not load embeddings model. Check installation (sentence-transformers, torch).")
            else:
                st.session_state.embeddings_model = embeddings_model
                prompt_parser_tuple = create_rag_prompt_template_with_parser()
                if not prompt_parser_tuple:
                    st.error("Setup failed: Could not create RAG prompt template or parser.")
                else:
                    st.session_state.rag_prompt_template, st.session_state.output_parser = prompt_parser_tuple
                    st.session_state.setup_complete = True
                    st.success("Setup complete! Ready to process documents.")
                    st.rerun() # Rerun to reflect the completed setup state

# Display setup status message if already complete
if st.session_state.setup_complete:
    st.success("System ready: API Key, LLM, Embeddings, and Prompts initialized.")


# --- Document Upload and Processing ---
# Disable uploaders if setup failed
upload_disabled = not st.session_state.setup_complete

st.markdown("---")
st.subheader("Upload Documents for Context")
col1, col2 = st.columns(2)

# --- Resume Processing ---
with col1:
    st.markdown("#### Resume")
    uploaded_resume = st.file_uploader(
        "Upload your Resume (PDF, DOCX)",
        type=['pdf', 'docx'],
        key="resume_uploader_widget",
        disabled=upload_disabled
    )

    # Check if a new file is uploaded or if the current one needs processing
    process_resume = False
    if uploaded_resume is not None:
        if uploaded_resume.name != st.session_state.current_resume_filename:
            st.info(f"New resume '{uploaded_resume.name}' detected. Ready for processing.")
            # Reset state for the new file
            st.session_state.resume_docs = None
            st.session_state.resume_processed = False
            st.session_state.resume_retriever = None
            st.session_state.current_resume_filename = uploaded_resume.name
            process_resume = True
        elif st.session_state.resume_processed:
             st.success(f"Resume '{uploaded_resume.name}' is loaded and ready.")
        # Allow reprocessing if it failed previously but file is still there
        elif not st.session_state.resume_processed:
             process_resume = True

    elif uploaded_resume is None and st.session_state.current_resume_filename is not None:
        # Handle file removal by user
        st.info(f"Resume '{st.session_state.current_resume_filename}' removed.")
        st.session_state.resume_docs = None
        st.session_state.resume_processed = False
        st.session_state.resume_retriever = None
        st.session_state.current_resume_filename = None

    # Perform processing if flagged and embeddings model is ready
    if process_resume and st.session_state.embeddings_model:
        with st.spinner(f"Processing Resume '{uploaded_resume.name}'..."):
            docs = load_and_parse_document(uploaded_resume)
            if docs:
                st.session_state.resume_docs = docs # Store loaded docs
                chunks = chunk_documents(docs)
                if chunks:
                    retriever = create_vector_store_retriever(chunks, st.session_state.embeddings_model)
                    if retriever:
                        st.session_state.resume_retriever = retriever
                        st.session_state.resume_processed = True
                        st.success(f"Resume '{uploaded_resume.name}' processed successfully.")
                        st.rerun() # Update UI status immediately
                    else:
                        st.error("Failed to create resume vector store.")
                        st.session_state.resume_processed = False # Ensure state reflects failure
                else:
                    st.error("Failed to chunk resume documents.")
                    st.session_state.resume_processed = False
            else:
                st.error(f"Failed to load or parse '{uploaded_resume.name}'.")
                st.session_state.resume_processed = False
                st.session_state.current_resume_filename = None # Reset filename if load failed

    elif process_resume and not st.session_state.embeddings_model:
        # This case should be less likely if upload is disabled when setup fails
        st.error("Cannot process resume: Embeddings model not available.")


# --- Job Description Processing ---
with col2:
    st.markdown("#### Job Description")
    uploaded_jd = st.file_uploader(
        "Upload Job Description (PDF, TXT, DOCX)",
        type=['pdf', 'txt', 'docx'],
        key="jd_uploader_widget",
        disabled=upload_disabled
    )

    process_jd = False
    if uploaded_jd is not None:
        if uploaded_jd.name != st.session_state.current_jd_filename:
            st.info(f"New JD '{uploaded_jd.name}' detected. Ready for processing.")
            st.session_state.jd_docs = None
            st.session_state.jd_processed = False
            st.session_state.jd_retriever = None
            st.session_state.key_skills = None # Reset skills for new JD
            st.session_state.current_jd_filename = uploaded_jd.name
            process_jd = True
        elif st.session_state.jd_processed:
            st.success(f"JD '{uploaded_jd.name}' is loaded and ready.")
        elif not st.session_state.jd_processed:
            process_jd = True

    elif uploaded_jd is None and st.session_state.current_jd_filename is not None:
        st.info(f"JD '{st.session_state.current_jd_filename}' removed.")
        st.session_state.jd_docs = None
        st.session_state.jd_processed = False
        st.session_state.jd_retriever = None
        st.session_state.key_skills = None # Clear skills
        st.session_state.current_jd_filename = None

    # Perform processing if flagged and prerequisites met (embeddings & LLM for skills)
    if process_jd and st.session_state.embeddings_model and st.session_state.llm:
        with st.spinner(f"Processing JD '{uploaded_jd.name}'..."):
            docs = load_and_parse_document(uploaded_jd)
            if docs:
                st.session_state.jd_docs = docs
                jd_full_text = "\n\n".join([doc.page_content for doc in docs])

                # Extract Key Skills (call function from agent_logic)
                skills = None
                if jd_full_text and st.session_state.key_skills is None:
                     with st.spinner("Extracting key skills from JD..."):
                         skills = extract_key_skills(jd_full_text, st.session_state.llm)
                         if skills:
                             st.session_state.key_skills = skills
                             # Don't show success here, wait for full processing
                         else:
                             st.warning("Could not extract key skills from JD.") # UI feedback

                # Chunking and Vector Store Creation (call functions from doc_processor)
                chunks = chunk_documents(docs)
                if chunks:
                    retriever = create_vector_store_retriever(chunks, st.session_state.embeddings_model)
                    if retriever:
                        st.session_state.jd_retriever = retriever
                        st.session_state.jd_processed = True
                        st.success(f"JD '{uploaded_jd.name}' processed successfully.") # UI feedback
                        st.rerun() # Update UI status
                    else:
                        st.error("Failed to create JD vector store.") # UI feedback
                        st.session_state.jd_processed = False
                        st.session_state.key_skills = None # Clear skills if retriever fails
                else:
                    st.error("Failed to chunk JD documents.") # UI feedback
                    st.session_state.jd_processed = False
                    st.session_state.key_skills = None # Clear skills if chunking fails
            else:
                st.error(f"Failed to load or parse '{uploaded_jd.name}'.") # UI feedback
                st.session_state.jd_processed = False
                st.session_state.key_skills = None
                st.session_state.current_jd_filename = None

    elif process_jd and (not st.session_state.embeddings_model or not st.session_state.llm):
         # This case should be less likely if upload is disabled when setup fails
         st.error("Cannot process JD: Embeddings model or LLM not available.")


# --- Display Extracted Skills ---
if st.session_state.get('key_skills'):
     st.markdown("---")
     with st.expander("💡 Key Skills Identified in Job Description (AI Extraction)", expanded=False):
         st.markdown(st.session_state.key_skills)


# --- User Inputs for Email Generation ---
st.markdown("---")
st.subheader("Email Parameters")
position = st.text_input("Position Applying For:", key="position_input", disabled=upload_disabled)
company = st.text_input("Company Name:", key="company_input", disabled=upload_disabled)
tone_options = ["Professional", "Enthusiastic", "Formal", "Friendly", "Direct"]
tone = st.selectbox("Select Email Tone:", tone_options, key="tone_select", disabled=upload_disabled)
notes = st.text_area("Additional Notes (e.g., specific skills to highlight, contact person):", key="notes_area", disabled=upload_disabled)


# --- Display Area for Generated Email ---
st.markdown("---")
st.subheader("Generated Email Output")

# Editable fields linked to session state for persistence and updates.
edited_subject = st.text_input(
    "Subject:",
    value=st.session_state.generated_subject,
    key="subj_display",
    disabled=upload_disabled
)
edited_body = st.text_area(
    "Email Body (Editable):",
    value=st.session_state.generated_body,
    height=400,
    key="body_display",
    disabled=upload_disabled
)

# Update session state from edits (Streamlit handles this via keys).
# This ensures the copy button gets the latest edited version.
st.session_state.generated_subject = edited_subject
st.session_state.generated_body = edited_body


# --- Action Buttons (Generate & Copy) ---
col_actions1, col_actions2 = st.columns([1, 5]) # Adjust ratio as needed

with col_actions1:
    # Generate Button
    generate_button_disabled = not st.session_state.setup_complete # Disable if setup failed
    if st.button("✨ Generate / Re-Generate", key="generate_button", disabled=generate_button_disabled, use_container_width=True):
        # Input validation
        if not position or not company:
            st.warning("Please enter the Position and Company Name.")
        # Check if RAG prerequisites are met.
        elif not st.session_state.resume_processed or not st.session_state.jd_processed:
             st.error("Cannot generate: Please ensure both Resume and Job Description are uploaded and processed successfully.")
        elif not st.session_state.rag_prompt_template or not st.session_state.output_parser:
             st.error("Cannot generate: RAG prompt template or parser not initialized.")
        else:
            # Call the generation logic from agent_logic
            with st.spinner("Synthesizing email using RAG..."): # UI feedback
                subject, body = generate_rag_email(
                    st.session_state.llm,
                    st.session_state.rag_prompt_template,
                    st.session_state.output_parser,
                    position, company, tone, notes,
                    st.session_state.get('resume_retriever'),
                    st.session_state.get('jd_retriever')
                )

            # Handle results in UI
            if subject is not None and body is not None:
                st.session_state.generated_subject = subject
                st.session_state.generated_body = body
                st.success("Email generated successfully!") # UI feedback
                st.rerun() # Update the display fields
            else:
                st.error("Email generation failed. Check console logs or try again.") # UI feedback

with col_actions2:
    # Copy Button
    if st.button("📋 Copy Email to Clipboard", key="copy_button", use_container_width=True, disabled=upload_disabled):
        # Use current values from the editable fields.
        if not edited_subject and not edited_body:
            st.warning("Nothing to copy. Please generate an email first.") # UI feedback
        else:
            full_email_text = f"Subject: {edited_subject}\n\n{edited_body}"
            try:
                pyperclip.copy(full_email_text)
                st.success("Email (Subject + Body) copied to clipboard!") # UI feedback
            except ImportError:
                 st.warning("Could not automatically copy. Install 'pyperclip' (`pip install pyperclip`) or copy manually.") # UI feedback
            except Exception as e:
                 # Handle common pyperclip error in headless environments.
                 if "clipboard mechanism" in str(e):
                      st.warning("Clipboard functionality may not be available in this environment. Please copy manually.") # UI feedback
                      st.code(full_email_text) # Show text for manual copy.
                 else:
                      st.error(f"Clipboard copy failed: {e}") # UI feedback

# --- Footer or Debug Info ---
# Optional: Add a footer or debug info
# st.markdown("---")
# with st.expander("Debug: Session State"): st.json(st.session_state)

