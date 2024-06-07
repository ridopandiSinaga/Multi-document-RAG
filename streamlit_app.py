import pdfplumber
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # Extract text from the PDF file
        with pdfplumber.open(uploaded_file) as pdf:
            documents = [page.extract_text() for page in pdf.pages]
            
        # Join the text from all pages into a single string
        document_text = "\n".join(documents)
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents([document_text])
        
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        
        # Create retriever interface
        retriever = db.as_retriever()
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        
        return qa.run(query_text)

# Page title
st.set_page_config(page_title='📄🔍 Ask the Document App')
st.title('📄🔍 Ask From Document')

# Explanation of the App
st.header('About the App')
st.write("""
The App is an question-answering platform that allows users to upload text documents and receive answers to their queries based on the content of these documents. Utilizing RAG approach powered by OpenAI's GPT models, the app provides insightful and contextually relevant answers.
""")

# Collapsible section for "How It Works"
with st.expander("How It Works?"):
    st.write("""
    - **Upload a Document**: You can upload any text document in `.pdf` format.
    - **Ask a Question**: After uploading the document, type in your question related to the document's content.
    - **Upload you OpenAI API key**.
    - **Get Answers**: AI analyzes the document and provides answers based on the information contained in it.
    """)

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')

# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)



# # Instructions for getting an OpenAI API key
# st.subheader("Get an OpenAI API key")
# st.write("You can get your own OpenAI API key by following the instructions:")
# st.write("""
# 1. Go to [OpenAI API Keys](https://platform.openai.com/account/api-keys).
# 2. Click on the `+ Create new secret key` button.
# 3. Next, enter an identifier name (optional) and click on the `Create secret key` button.
# """)

# Collapsible section for "Get an OpenAI API Key"
with st.expander("How I Get an OpenAI API Key?"):
    st.subheader("Instructions for getting an OpenAI API key")
    st.write("You can get your own OpenAI API key by following the instructions:")
    st.write("""
    1. Go to [OpenAI API Keys](https://platform.openai.com/account/api-keys).
    2. Click on the `+ Create new secret key` button.
    3. Next, enter an identifier name (optional) and click on the `Create secret key` button.
    """)