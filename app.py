import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # model_name = "hkunlp/instructor-xl"
    # model_kwargs = {'device': 'cpu'}
    # encode_kwargs = {'normalize_embeddings': True}
    # hf = HuggingFaceInstructEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )
    # embeddings = hf
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature": 0.5, "max_length": 512})

    # llm = HuggingFaceHub(repo_id="microsoft/DialoGPT-medium", model_kwargs={"temperature":0.5, "max_length":512})
    
    # llm = HuggingFaceHub(repo_id="describeai/gemini", model_kwargs={"temperature": 0.5, "max_length": 512})


    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


# def main():
#     load_dotenv()
    
#     st.set_page_config(page_title="Chat with multiple PDFs",
#                        page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get pdf text
#                 raw_text = get_pdf_text(pdf_docs)

#                 # get the text chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(
#                     vectorstore)
                
#     # Add stylish footer
#     st.markdown("""
#         <style>
#             .footer {
#                 position: fixed;
#                 bottom: 0;
#                 width: 50%;
#                 background-color: #f1f1f1;
#                 text-align: center;
#                 padding: 10px 0;
#                 font-size: 16px;
#                 color: #4c4c4c;
#                 border-top: 1px solid #e0e0e0;
#             }
#             .footer a {
#                 color: #4c4c4c;
#                 text-decoration: none;
#                 font-weight: bold;
#             }
#             .footer a:hover {
#                 text-decoration: underline;
#             }
#         </style>
#         <div class="footer">
#             <p>Developed by <a href="#">Sai</a></p>
#         </div>
#     """, unsafe_allow_html=True)

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    
    custom_css = """
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .main {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #1e1e1e;
        text-align: center;
        padding: 10px 0;
        font-size: 16px;
        color: #ffffff;
        border-top: 1px solid #333333;
    }
    .footer a {
        color: #ffffff;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
    st.markdown("""
        <div class="footer">
            <p>Developed by <a href="#">Sai</a></p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()