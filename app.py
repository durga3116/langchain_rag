import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader


f = open(r"C:\Users\hp\OneDrive\Desktop\New folder\keys\.google_gemini_api_key.txt")
api_key = f.read()

st.title(":green[ PDF Search:Using Advanced RAG System ] 🤖📄")
st.subheader("📑🔍[PDF Navigation with RAG Technology Using Langchain & LLM]:orange")


user_input = st.text_area("👉[Ask anything from Leave No Context behind paper]:yellow")



chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-pro-latest")

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a helpful assistant, trained to provide correct answers
    Your responses should be in correct format """),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Question: {question}

Answer: """)
])

output_parser = StrOutputParser()


loader = PyPDFLoader("paper.pdf")
pages = loader.load_and_split()

data = loader.load()

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key,
                                               model="models/embedding-001")


db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
db.persist()

# Setting a Connection 
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a helpful assistant, trained to provide correct answers based on the context provided.
    Your answers should be correct format """),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Context:
{context}

Question:
{question}

Answer: """)
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

if st.button("Generate Response"):
    response = rag_chain.invoke(user_input)
    st.write(response)






