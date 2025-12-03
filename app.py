import streamlit as st
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# --- Page Setup ---
st.set_page_config(page_title="Capex Detective", layout="wide")

st.title("🏭 Capex Detective: AI Equity Research Agent")
st.markdown("""
**Objective:** Automated extraction of Capex Guidance and Capacity Expansion plans from Annual Reports.
\n*Built with LangChain & OpenAI GPT-4o-mini*
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    # Option 1: User enters key (Best for public demos so you don't pay)
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    
    # Option 2: Hardcode it (Only do this if using Streamlit Secrets for the CV link)
    # if 'OPENAI_API_KEY' in st.secrets:
    #     api_key = st.secrets['OPENAI_API_KEY']
    
    st.divider()
    st.markdown("### How to use")
    st.markdown("1. Enter API Key.")
    st.markdown("2. Upload PDF (Annual Report).")
    st.markdown("3. Click 'Run Analysis'.")

# --- Main App ---
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")

    if uploaded_file is not None:
        st.info("Reading PDF... parsing financial data.")
        
        # 1. Process PDF
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # 3. Embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        st.success("PDF Processed! AI is ready.")
        st.divider()
        
        # 4. Analysis
        questions = [
            "What is the total Capital Expenditure (Capex) guidance for the upcoming fiscal years? (Look for ₹ or USD amounts)",
            "What are the specific capacity expansion targets? (Look for units like MTPA, Units/Year)",
            "Summarize the key R&D initiatives and the R&D spend as a % of sales."
        ]
        
        if st.button("Run Analyst Scan"):
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
            chain = load_qa_chain(llm, chain_type="stuff")
            
            for q in questions:
                st.markdown(f"#### {q}")
                docs = knowledge_base.similarity_search(q)
                response = chain.run(input_documents=docs, question=q)
                st.write(response)
                st.markdown("---")

elif not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
