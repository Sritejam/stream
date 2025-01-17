import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import streamlit as st
from typing import List
import re
from langchain.schema import Document
from langchain.document_loaders import TextLoader

class HeadingTextSplitter:
    def __init__(self, heading_pattern: str = r"^####\s.*", max_chunk_size: int = 1000):
        """
        Splits text into chunks based on headings.

        Args:
            heading_pattern (str): Regex pattern to identify headings. Defaults to lines starting with "####".
            max_chunk_size (int): Maximum chunk size to avoid excessively large chunks.
        """
        self.heading_pattern = re.compile(heading_pattern, re.MULTILINE)
        self.max_chunk_size = max_chunk_size

    def split_text(self, text: str) -> List[Document]:
        """
        Splits the text into chunks based on headings and ensures chunks are not overly large.

        Args:
            text (str): Input text.

        Returns:
            List[Document]: List of `Document` objects containing the split text.
        """
        chunks = []
        current_chunk = []
        current_chunk_size = 0

        lines = text.split("\n")
        for line in lines:
            if self.heading_pattern.match(line) and current_chunk_size > 0:
                # Save the current chunk when a new heading is encountered
                chunks.append(Document(page_content="\n".join(current_chunk)))
                current_chunk = []
                current_chunk_size = 0

            # Add the line to the current chunk
            current_chunk.append(line)
            current_chunk_size += len(line)

            # Save chunk if it exceeds the max size
            if current_chunk_size > self.max_chunk_size:
                chunks.append(Document(page_content="\n".join(current_chunk)))
                current_chunk = []
                current_chunk_size = 0

        # Add the remaining content as the last chunk
        if current_chunk:
            chunks.append(Document(page_content="\n".join(current_chunk)))

        return chunks


# Load and split the document
document_loader = TextLoader("product-data.txt")
document = document_loader.load()[0].page_content  # Assuming one document is loaded

# Use the custom HeadingTextSplitter
text_splitter = HeadingTextSplitter(heading_pattern=r"^####\s.*", max_chunk_size=1000)
chunks = text_splitter.split_text(document)


# Load the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set in the environment variables.")
    raise ValueError("OpenAI API key is missing.")

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

# # Load and split the document
# document_loader = TextLoader("product-data.txt")
# document = document_loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = text_splitter.split_documents(document)

# Create a vector store with FAISS
vector_store = FAISS.from_documents(chunks, embeddings)
# retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retriever = vector_store.as_retriever()


# Load the custom prompt from secrets
CUSTOM_PROMPT = os.getenv("promptss")

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
    return (
        f"{CUSTOM_PROMPT}\n\n"
        f"Question: {question}\n\n"
    )
if question:
    # Call the chain with the current question and chat history
     # Format the prompt with the retrieved context and question
    question = apply_custom_prompt(CUSTOM_PROMPT, question)
    # response = rag_chain({"question": question, "chat_history": st.session_state.chat_history})
    response = rag_chain({"question": question})
    # Display the response
    st.write(response["answer"])
    
    # Update chat history
    st.session_state.chat_history.append((question, response["answer"]))
