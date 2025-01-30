import streamlit as st  
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA, StuffDocumentsChain
from langchain.memory import ConversationBufferMemory  
import base64  
# Streamlit UI
st.title("📄 PDF Chatbot with AI")
st.sidebar.header("Upload a PDF")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

# Hugging Face API Token
HUGGINGFACEHUB_API_TOKEN = "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"
if uploaded_file is not None:
    # Convert PDF file to base64
    base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")

    # Embed PDF in an iframe
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%" height="600px"
            type="application/pdf">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

if uploaded_file:
    # Save PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load PDF text
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    #st.markdown(docs)

    # Ensure HuggingFace embeddings work correctly
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Embedding model error: {e}")
        st.stop()

    # Split text into semantic chunks
    text_splitter = SemanticChunker(embeddings)
    documents = text_splitter.split_documents(docs)

    # Generate vector embeddings and store in FAISS
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 chunks

    # Load Ollama LLM
    llm = Ollama(model="deepseek-r1:1.5b")

    # Prompt Template
    prompt = """
    1. Use ONLY the context below.
    2. If unsure, say "I don’t know".
    3. Keep answers under 4 sentences.

    Context: {context}

    Question: {question}

    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # Chat memory to maintain history
    memory = ConversationBufferMemory(input_key="question", memory_key="chat_history")

    # Chain 1: Generate answers
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, memory=memory)

    # Chain 2: Combine document chunks
    document_prompt = PromptTemplate(
        template="Context:\ncontent:{page_content}\nsource:{source}",
        input_variables=["page_content", "source"]
    )

    # Final RAG pipeline
    qa = RetrievalQA(
        combine_documents_chain=StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name="context"
        ),
        retriever=retriever
    )

    # Chatbot UI
    st.subheader("💬 Ask Questions About Your PDF")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question about the PDF...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = qa.invoke(user_input)["result"]

        # Store messages
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
