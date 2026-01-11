# Import dependencies
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables from .env file
load_dotenv()

# Streamlit app settings
st.set_page_config(page_title="ASKSIDEWAYS", page_icon=":bar_chart:")

chunk_size = int(os.getenv('chunk_size', 1000))

# Show file upload section
st.write("Upload a PDF document")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_file:
    if st.button("Submit the file"):
        with st.spinner("Uploading and processing document..."):
            # Save the uploaded file temporarily
            with open("uploaded_pdf.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load and split the PDF content
            loader = PyPDFLoader("uploaded_pdf.pdf")
            pages = loader.load_and_split()

            # Split the content into chunks for readability
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
            documents = text_splitter.split_documents(pages)

            # Display the extracted text
            st.write("Extracted Text from PDF:")
            for doc in documents:
                st.write(doc.page_content)


