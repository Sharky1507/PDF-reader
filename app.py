import streamlit as st
import os

# Update imports to use langchain-community
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Import Gemini components
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Page configuration
st.set_page_config(page_title="Document Q&A with RAG", layout="wide")
st.title("Document Q&A with RAG")
st.write("Ask questions about your PDF documents using Retrieval Augmented Generation")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Google API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

# Check if API key is provided
if not api_key:
    st.warning("Please enter your Google API key in the sidebar.")
    st.stop()

# Set API key
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
    st.session_state.qa_chain = None

# Process uploaded document
if uploaded_file is not None:
    with st.spinner("Processing document..."):
        try:
            # Save uploaded file temporarily
            with open("temp_document.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load document
            loader = PyPDFLoader("temp_document.pdf")
            documents = loader.load()
            
            if not documents:
                st.error("No content could be extracted from the PDF. Please check if the PDF contains text.")
                st.stop()
                
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            
            if not docs:
                st.error("Document splitting resulted in no chunks. The PDF might be empty or contain only images.")
                st.stop()
                
            st.info(f"Document processed into {len(docs)} chunks")
            
            # Create embeddings and vector store with Gemini
            # Updated to use the correct embedding model name
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")            
            # Test a single embedding to make sure it works
            test_text = docs[0].page_content
            try:
                test_embedding = embeddings.embed_query(test_text[:100])
                st.info(f"Embedding test successful: Vector dimension = {len(test_embedding)}")
            except Exception as e:
                st.error(f"Embedding test failed: {str(e)}")
                st.stop()
            
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            # Create retriever
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Create custom prompt
            custom_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
            You are an AI assistant. Use the context below to answer the question.
            If you don't know the answer, just say you don't know. Don't make anything up.

            Context: {context}
            Question: {question}

            Answer:"""
            )
            
            # Create QA Chain with Gemini LLM
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2),
                chain_type="stuff",
                retriever=st.session_state.retriever,
                chain_type_kwargs={"prompt": custom_prompt}
            )
            
            st.success("Document processed successfully!")
        
        except Exception as e:
            st.error(f"An error occurred: {type(e).__name__}: {str(e)}")
            st.stop()

# User interface for questions
st.header("Ask a question about your document")

# Only show question input if document is processed
if st.session_state.retriever is not None:
    query = st.text_input("Your question:", "What are the key takeaways from the document?")
    
    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            try:
                response = st.session_state.qa_chain.run(query)
                st.subheader("Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
else:
    if uploaded_file is None:
        st.info("Please upload a PDF document to get started.")