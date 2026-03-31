# os - for environment variables
import os
# sys - for exiting on error
import sys
# dotenv - for loading .env file
from dotenv import load_dotenv
# langchain_google_genai - for LLM and embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# langchain_community - for document loading, text splitting, and vector store
from langchain_community.document_loaders import PyPDFLoader
# langchain_text_splitters - for splitting documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
# langchain_community.vectorstores - for Chroma vector store
from langchain_community.vectorstores import Chroma

# Store your API Key in a .env file with the line: GOOGLE_API_KEY="your_api_key_here"
# Load your API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Validate API Key
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    sys.exit(1)

# Import RetrievalQA
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    print("Error: Run 'pip install langchain-classic'")
    sys.exit(1)

# Function to get embeddings model
def get_embeddings():
    candidate_models = [
        "models/gemini-embedding-001", 
        "models/gemini-embedding-2-preview",
    ]

    # Track the last error for better debugging, last_error is used to provide more context in the final error message if all models fail
    last_error = None

    # Try each model in the list until one works, if a model fails, catch the error and move to the next one, if all models fail, raise an error with the last encountered issue    
    for model_name in candidate_models:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
            # Short test to confirm connectivity
            embeddings.embed_query("test")
            return embeddings, model_name
        except Exception as err:
            last_error = err
            print(f"Skipping {model_name}: {err}")

    raise RuntimeError(
        f"Initialization failed. Final error: {last_error}. "
        "Check your GOOGLE_API_KEY and Quota in Google AI Studio."
    )


def chatbot(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Split documents into chunks of 800 characters with an overlap of 80 characters
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    texts = splitter.split_documents(docs)

    # Remove empty chunks that may have been created during splitting
    texts = [t for t in texts if t.page_content.strip()]

    # Get embeddings model and print which one is being used
    embeddings, model_info = get_embeddings()
    print(f"Using: {model_info}")

    # Initialize Chroma
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)

    # Initialize the LLM with the specified model and temperature
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.7,
            api_key=api_key
        )

    # Create the RetrievalQA chain using the LLM and the vector store as the retriever
    # chain_type="stuff" -> This means that the retrieved chunks will be concatenated together and fed into the LLM as a single input.
    # "k": 3 -> top 3 most relevant chunks will be retrieved from the vector store for each query. 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    return qa_chain

if __name__ == "__main__":
    if os.path.exists("./sample.pdf"):
        try:
            bot = chatbot("./sample.pdf")
            print("\n--- Chatbot Ready ---\n")
            while True:
                user_input = input("Ask about the PDF (or 'exit'): ")
                if user_input.lower() in ['exit', 'quit']: break
                
                # Invoke the bot with the user's query and print the answer
                response = bot.invoke({"query": user_input})
                print(f"\nAnswer: {response['result']}\n")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("File 'sample.pdf' not found.")
