import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os   

    data = ""
    for pdf in file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            data += page.extract_text()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # chunks = text_splitter.split_documents(data)
    chunks = text_splitter.split_text(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    #vector_store = Chroma.from_documents(chunks, embeddings)
    vector_store  = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=5):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})


    prompt_template = """You are examining a document. Use only the following piece of context to answer the questions at the end. 
    If you don't know the answer, just say that you don't know in a polite way, don't try to make up an answer. 
    If the context is not relevant, do not answer the question by using your own knowledge about the topic
    Do not add any observations or comments. Answer only in Vietnamese, do add a polite sentence to ask our customer if they would like more information".
    
    CONTEXT {context}

    QUESTION: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}



    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,chain_type_kwargs=chain_type_kwargs)

    answer = chain.run(q)
    return answer


# calculate embedding cost using tiktoken



# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.image('img.png')
    st.subheader('Chat with multiple PDFs :books:')

    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload pdf files:', type=['pdf'], accept_multiple_files=True)

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=1000, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=5, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
             

                data = load_document(uploaded_file)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

            

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {q} \nA: {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)

# run the app: streamlit run ./chat_with_documents.py

