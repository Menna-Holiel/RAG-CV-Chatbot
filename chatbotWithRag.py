from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_together import ChatTogether
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

#load api key
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

app = FastAPI()

CVS_dir = "CVs"

# 1) load cvs
def load_cvs(directory_path):
    documents =[]
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory_path, file_name))
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = file_name   # attach CV name
            documents.extend(docs)
    return documents        


# 2) split documents into chunks
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# 3) embed & create vector store
def create_vectorstore(docs):
    # embedding = OpenAIEmbeddings(open_api_key=api_key)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore

# Indexing 
documents = load_cvs(CVS_dir)
chunks = split_docs(documents)
vectorstore = create_vectorstore(chunks)

# 4) retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5) llm
llm = ChatTogether(
    model = "meta-llama/Llama-3-70b-chat-hf",
    together_api_key = api_key,
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. 
If the user provides a job description, use the CVs to answer. 
If it’s a general question, just chat normally.

Conversation so far:
{chat_history}

User request:
{job_description}

Relevant CV context:
{context}

Answer:
""")

memory = ConversationBufferMemory(memory_key="chat_history",
                                   input_key="job_description", 
                                  return_messages=True)
chain = LLMChain(
    llm=llm,
     prompt=prompt,
    memory=memory,
    output_parser=StrOutputParser()
)
# chain = prompt | llm | StrOutputParser()

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_llama(request: ChatRequest):
    # retrieve relevent cv chunks
    docs = retriever.get_relevant_documents(request.prompt)
    context = "\n".join([f"[{d.metadata['source']}] {d.page_content}" for d in docs]) or "No relevant CVs found."

    answer = chain.predict(
        job_description=request.prompt,
        context=context
    )

    return ChatResponse(answer = answer)