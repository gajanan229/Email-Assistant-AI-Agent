# AI Job Application Email Assistant

A Streamlit web application that leverages Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to help users draft personalized job application emails based on their resume and a specific job description.

## Features

*   **Document Upload:** Accepts user's resume (PDF, DOCX) and job description (PDF, TXT, DOCX).
*   **Contextual Processing:** Loads, chunks, and embeds document content using Langchain and Sentence Transformers.
*   **Vector Store:** Creates a FAISS vector store for efficient similarity search.
*   **RAG Implementation:** Retrieves relevant context from both the resume and job description based on the target role and company.
*   **LLM-Powered Generation:** Uses OpenAI's LLM with a structured prompt (including retrieved context) to generate tailored email drafts.
*   **Customization:** Allows users to specify the target position, company, desired email tone, and add custom notes.
*   **Structured Output:** Parses the LLM response to separate the email subject and body.
*   **Key Skill Extraction:** Optionally extracts key skills/requirements directly from the job description using an LLM call.
*   **Editable Output:** Displays the generated subject and body in editable fields within the Streamlit UI.
*   **Copy to Clipboard:** Provides a button to easily copy the final email content.

## Tech Stack

*   **Language:** Python 3.x
*   **Framework:** Streamlit (for the web UI)
*   **AI/LLM Orchestration:** Langchain
    *   **Document Loaders:** `PyPDFLoader`, `TextLoader`, `UnstructuredFileLoader`
    *   **Text Splitters:** `RecursiveCharacterTextSplitter`
    *   **Embeddings:** `SentenceTransformerEmbeddings` (from `langchain-community`)
    *   **Vector Stores:** `FAISS` (from `langchain-community`)
    *   **Retrievers:** FAISS `as_retriever`
    *   **LLMs:** `ChatOpenAI` (from `langchain-openai`, using `gpt-4o` model)
    *   **Prompts:** `ChatPromptTemplate`, `SystemMessagePromptTemplate`, `HumanMessagePromptTemplate`
    *   **Output Parsers:** `StructuredOutputParser`
*   **Environment Variables:** `python-dotenv`
*   **Clipboard:** `pyperclip`

## Installation & Setup

Follow these steps to set up and run the project locally:

1.  **Clone the Repository:**
    ```powershell
    git clone <your-repository-url>
    cd Email-Assistant-AI-Agent
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```powershell
    # Create the virtual environment
    python -m venv .venv

    # Activate the virtual environment (PowerShell)
    .\.venv\Scripts\Activate.ps1
    # For Git Bash / Linux / macOS: source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Create `.env` File:**
    Create a file named `.env` in the root directory of the project (`Email-Assistant-AI-Agent/`).

5.  **Add OpenAI API Key:**
    Open the `.env` file and add your OpenAI API key:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    *   **Important:** You need a valid OpenAI API key with available credits for the application to work.

## Usage

1.  **Run the Streamlit Application:**
    Make sure your virtual environment is activated. Then run:
    ```powershell
    streamlit run app.py
    ```
    This will open the application in your web browser.

2.  **Application Workflow:**
    *   The application will first initialize the necessary components (LLM, embeddings).
    *   Upload your resume file using the "Resume" uploader.
    *   Upload the job description file using the "Job Description" uploader.
    *   Wait for the application to process both documents (indicated by success messages).
    *   Optionally review the extracted key skills from the job description.
    *   Fill in the "Position Applying For", "Company Name", select an "Email Tone", and add any "Additional Notes".
    *   Click the "✨ Generate / Re-Generate" button.
    *   The generated email subject and body will appear in the editable fields below.
    *   Review and edit the generated content as needed.
    *   Click the "📋 Copy Email to Clipboard" button to copy the final subject and body.

## Project Structure

```text
Email-Assistant-AI-Agent/
│
├── app.py                 # Main Streamlit UI and application flow control
├── requirements.txt       # Project dependencies
├── README.md              # This file
├── .env                   # For API keys (Not committed to Git)
└── src/                   # Source code directory for backend logic
    ├── __init__.py        # Makes 'src' a Python package
    ├── agent_logic.py     # Core Langchain/RAG generation, LLM config, prompting, parsing
    └── doc_processor.py   # Document loading, chunking, embedding, vector store creation


*   **`app.py`**: Handles the user interface elements, state management, and orchestrates calls to the backend logic.
*   **`src/doc_processor.py`**: Contains functions responsible for loading, parsing, chunking documents, and creating the vector store retriever.
*   **`src/agent_logic.py`**: Contains functions for configuring the LLM, managing prompts, performing RAG generation, extracting skills, and parsing outputs.
```

