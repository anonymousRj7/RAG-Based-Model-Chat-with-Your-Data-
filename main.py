import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import shutil
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



# Set Streamlit page configuration and header
st.set_page_config("Chat PDF")
st.title("")
st.markdown("<h1 style='text-align: center;'>Project 1 </h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>QUERY ON YOUR DATA WITH GEMINI ♊</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center;'>Enter Your Question ...</h1>", unsafe_allow_html=True)
with st.sidebar:
    # Input field 
    # st.text("Press the Submit button to upload a file.")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True,type=['pdf'])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


    if st.button("Submit"):
        with st.spinner("Processing..."):
            raw_text = ""
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            text_chunks = text_splitter.split_text(raw_text)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")






st.markdown("<h5 style='text-align: center;'>Press Ctrl+Enter to get your response</h1>", unsafe_allow_html=True)
user_question  = st.text_area(label="Enter your Query...")
if user_question :
            
  # Define prompt template for conversational chain
  prompt_template = """
Given the context information and not prior knowledge, 
answer the query asking about citations over different topics.

                Context:
                {context}

                Question:
                {question}

                Answer:

            """
  # Initialize ChatGoogleGenerativeAI model
  model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, safety_settings = {
                                  HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE, 
                                  # HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
                                  # HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
                                  # HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
                                  # HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,

})



            
  prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
  chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
  docs = new_db.similarity_search(user_question)
  response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)





  #Display response
  st.write("Reply: ", response["output_text"])

