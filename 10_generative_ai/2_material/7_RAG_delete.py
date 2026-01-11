# Import dependencies
import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from google import genai  

# Load environment variables from .env file
load_dotenv()

# Set API keys from environment variables
gemni_api_key = os.getenv('GEMINI_API')
api_key = os.getenv('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = api_key

# Define constants from environment variables
namespace = os.getenv('namespace', 'wondervector5000')
index_name = os.getenv('index_name', 'class1')
chunk_size = int(os.getenv('chunk_size', 1000))



USERNAME ="Farhan"
PASSWORD = "123456"

# Set up embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
docsearch = PineconeVectorStore.from_documents(
    documents="", 
    index_name=index_name, 
    embedding=embeddings, 
    namespace=namespace
)
time.sleep(1)

# Set up Google Genai client
client = genai.Client(api_key=gemni_api_key)

# Function to add background color
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #a0bfb9;
            color: #895051;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )



# Streamlit app settings
st.set_page_config(page_title="ASKSIDEWAYS", page_icon=":bar_chart:")
add_bg_from_url()

# Session state for login, feedback, question, and visibility
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login logic
if not st.session_state.logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# Main app logic (after login)
else:
    st.title("Envoy")

    # Check if Pinecone database is empty

    # Show file upload section only if the database is empty
    st.write("Upload documents")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    if uploaded_file:
        if st.button("Submit the file"):
            with st.spinner("Uploading and processing document..."):
                with open("uploaded_pdf.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader("uploaded_pdf.pdf")
                pages = loader.load_and_split()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
                documents = text_splitter.split_documents(pages)
                docsearch = PineconeVectorStore.from_documents(
                    documents=documents,
                    index_name=index_name,
                    embedding=embeddings,
                    namespace=namespace,
                )
            st.success("Document uploaded and processed. You can now ask questions about its content.")


    # Question input and response
    question = st.text_input("Ask queries related to the uploaded knowledge:")
    if st.button("Submit query"):
        with st.spinner("Getting your answer..."):
            retrieved_docs = docsearch.as_retriever(search_kwargs={"k": 10}).get_relevant_documents(question)
            
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Create prompt with context and question
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Generate response using Google SDK
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            st.subheader("Answer")
            st.write(response.text)
            st.subheader("Context")
            st.write(context)


    # Clear database button
    if st.button("Clear the database"):
        with st.spinner("Clearing the database..."):
            try:
                pc = Pinecone(api_key=api_key)
                index = pc.Index(index_name)
                index.delete(delete_all=True, namespace=namespace)
                st.success("Database cleared!")
            except:
                st.error("The database is already empty.")


    