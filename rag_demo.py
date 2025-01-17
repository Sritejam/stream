import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import streamlit as st

# Load the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set in the environment variables.")
    raise ValueError("OpenAI API key is missing.")

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

# Load and split the document
document_loader = TextLoader("product-data.txt")
document = document_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)

# Create a vector store with FAISS
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

# Load the custom prompt from secrets
CUSTOM_PROMPT = os.getenv("promptss")
# # Define the simplified prompt
# promptss = (
#     "You are a professional assistant specialized in answering questions strictly about Sriteja Madishetty. "
#     "Use the provided context to respond positively, highlighting the benefits of employing him. "
#     "If the answer is not available in the context, acknowledge that you cannot answer and ask them to reach out to him. "
#     "Always provide links when giving contact information. Summarize and provide a neat answer."
# )


# Create the Conversational Retrieval Chain
rag_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

# Streamlit UI
st.title("Ask About Sriteja Madishetty")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Your Question:")

# Custom Prompt Application
def apply_custom_prompt(context, question):
    """Formats the custom prompt with the retrieved context and question."""
    return CUSTOM_PROMPT.format(context=context, question=question)


if question:
    # Call the chain with the current question and chat history
     # Format the prompt with the retrieved context and question
    question = apply_custom_prompt(CUSTOM_PROMPT, question)
    response = rag_chain({"question": question, "chat_history": st.session_state.chat_history})
    
    # Display the response
    st.write(response["answer"])
    
    # Update chat history
    st.session_state.chat_history.append((question, response["answer"]))
