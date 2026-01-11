# Import dependencies
import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader 
from google import genai
import uuid

# Load environment variables from .env file
load_dotenv()

# Set API keys from environment variables
gemni_api_key = os.getenv('GEMINI_API')
api_key = os.getenv('PINECONE_API_KEY')

# Define constants from environment variables
namespace = os.getenv('namespace', 'wondervector5000')
index_name = os.getenv('index_name', 'test1')
chunk_size = int(os.getenv('chunk_size', 1000))

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# Set up embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Google Genai client
client = genai.Client(api_key=gemni_api_key)



# Streamlit app settings
st.set_page_config(page_title="ASKSIDEWAYS", page_icon=":bar_chart:")
st.title("Envoy")



st.write("Upload documents")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_file and st.button("Submit the file"):
    with st.spinner("Uploading and processing document..."):
        with open("uploaded_pdf.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader("uploaded_pdf.pdf")
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
        documents = text_splitter.split_documents(pages)
        
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
        
        batch_size = 1000
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
        
        time.sleep(2)
        st.success(f"Document uploaded and processed. {len(vectors_to_upsert)} chunks indexed.")


question = st.text_input("Ask queries related to the uploaded knowledge:")
if st.button("Submit query"):
    with st.spinner("Getting your answer..."):
        query_embedding = embeddings.embed_query(question)
        results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        context = "\n\n".join([match.metadata.get("text", "") for match in results.matches])
        
        st.title("Context:")
        st.write(context)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        st.title("Answer:")
        st.write(response.text)



   