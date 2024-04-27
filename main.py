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
st.header("Chat with PDF using Gemini💁")

# Input field for user question
user_question = st.text_input("Ask a Question from the PDF Files")

# Check if user has entered a question
if user_question:
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    



    if st.button("Submit & Process"):
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



            
            # Define prompt template for conversational chain
            prompt_template = """
                Please provide a detailed answer based on the given context.
                Ensure that your response contains all relevant information.If the answer cannot be found in the provided context, simply state, "The answer is not available in the context.".
                Avoid providing incorrect information.

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



            # Display response
            st.write("Reply: ", response["output_text"])
