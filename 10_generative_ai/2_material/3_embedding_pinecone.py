# Import dependencies
import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

# Load environment variables from .env file
load_dotenv()

# Set API keys from environment variables
api_key = os.getenv('PINECONE_API_KEY')

# Define constants from environment variables
namespace = os.getenv('namespace', 'wondervector5000')
index_name = os.getenv('index_name', 'test1')
chunk_size = int(os.getenv('chunk_size', 1000))

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# Set up embeddings model (for manual embedding generation)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Streamlit app settings
st.set_page_config(page_title="ASKSIDEWAYS", page_icon=":bar_chart:")

st.write("Upload documents")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_file and st.button("Submit the file"):
    with st.spinner("Uploading and processing document..."):
        # Save uploaded file
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and split PDF
        loader = PyPDFLoader("uploaded_pdf.pdf")
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
        documents = text_splitter.split_documents(pages)
        
        # Generate embeddings and prepare vectors for Pinecone
        vectors_to_upsert = []
        for i, doc in enumerate(documents):
            embedding = embeddings.embed_query(doc.page_content)
            vector = {
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", i),
                    "source": doc.metadata.get("source", "uploaded_pdf.pdf")
                }
            }
            vectors_to_upsert.append(vector)
        
        # Batch upsert (max 1000 vectors per batch)
        batch_size = 1000
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
        
        time.sleep(2)
        st.success(f"Document uploaded and processed. {len(vectors_to_upsert)} chunks indexed in namespace '{namespace}'.")



