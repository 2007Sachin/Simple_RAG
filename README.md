# Simple_RAG
Simple RAG Chatbot
1. Simple RAG
Concept : The most basic form of RAG. It follows a linear flow: retrieve relevant documents from a vector database, then generate an answer using an LLM.
Steps :
User query → Retrieve top-k documents from the vector store.
Pass the query + retrieved documents to the LLM.
Generate a final answer.
Use Case : Suitable for straightforward QA tasks where the retrieval is assumed to be accurate
