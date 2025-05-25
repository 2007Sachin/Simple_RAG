import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from config import GROQ_API_KEY, MODEL_NAME, EMBEDDING_MODEL_NAME

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Simple RAG Chatbot")
st.title("📄 Simple RAG Chatbot")

# Load LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Load and process the PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load_and_split()
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Build QA Chain
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

    # Chat Interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = qa_chain.invoke(user_input)
            st.markdown(response["result"])
        st.session_state.chat_history.append({"role": "assistant", "content": response["result"]})