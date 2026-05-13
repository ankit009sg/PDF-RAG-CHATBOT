import streamlit as st
import pdfplumber
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.header("RAG Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions")
    if file is not None:
        if file.size > 5 * 1024 * 1024:
            st.error("File too large")
            st.stop()

# cache the embedding model so it loads only once
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

#Cache the vector store per file so it rebuilds only when file changes
@st.cache_resource
def build_vector_store(file_bytes):
    import io
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"



    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    embeddings = load_embeddings()
    return FAISS.from_texts(chunks, embeddings)

if file is not None:
    with st.spinner("Processing PDF..."):
        vector_store = build_vector_store(file.read())

    user_question = st.chat_input("Type your question here: ")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    #Similarity search to fetch top 3 chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024,
        api_key=st.secrets["GROQ_API_KEY"]  
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "strictly follow- validate query if it is about the context prceed further else reply type valid query" 
         "1. Provide complete well answered using context below.\n"
         "2. Include relevant details, numbers, and explanations to give thorough response.\n"
         "3. If the context mentions relevant information, include it to give the full picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge.\n"
         "5. Summarise long information, ideally in bullets where needed.\n"
         "6. If the information is not in the context, say so politely.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    #Streaming the response so text appears word-by-word instead of all at once
    if user_question:
        st.write_stream(chain.stream(user_question))