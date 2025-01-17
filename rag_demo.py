import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

# Load the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set in the environment variables.")
    raise ValueError("OpenAI API key is missing.")

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Load and split the document
document_loader = TextLoader("product-data.txt")
document = document_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(document)

# Create a vector store with FAISS
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

# Create the Conversational Retrieval Chain
rag_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

# Streamlit UI
st.title("Ask About Sriteja Madishetty")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Your Question:")

if question:
    # Call the chain with the current question and chat history
    response = rag_chain({"question": question, "chat_history": st.session_state.chat_history})
    
    # Display the response
    st.write(response["answer"])
    
    # Update chat history
    st.session_state.chat_history.append((question, response["answer"]))
