import os 
import tempfile
import uuid
from app_utils import reset_chat
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from tempfile import NamedTemporaryFile
from DocumentManager import DocumentManager
from EmbeddingManager import EmbeddingManager
from ConversationalRetrievalAgent import ConversationalRetrievalAgent

os.environ["LLAMA_PARSE_API_KEY"] = st.secrets["LLAMA_PARSE_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]



def main():
    if "id" not in st.session_state:
        st.session_state.id = uuid.uuid4()
        st.session_state.file_cache = {}

    st.set_page_config(layout="wide")

    with st.sidebar:
        uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile() as temp_file, st.status(
                "Processing document", expanded=False, state="running"
            ):
                with open(temp_file.name, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.write("Indexing...")
                st.write("Complete, ask your questions...")

    if uploaded_file is None:
        st.stop()

    with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
        f.write(uploaded_file.getvalue())
        doc_manager = DocumentManager()
        sections = doc_manager.split_documents(document_path=f.name)
    # Creation and persistence of embeddings
    embed_manager = EmbeddingManager(sections)
    embed_manager.create_and_persist_embeddings()

    # Setup and use of conversation bots
    bot = ConversationalRetrievalAgent(embed_manager.vectordb)
    bot.setup_bot()
    col0, col1, col2 = st.columns([6, 6, 1])
    with col0:
        binary_data = uploaded_file.getvalue()
        pdf_viewer(input=binary_data, width=700)


    with col1:
        # Initialize chat history
        if "messages" not in st.session_state:
            reset_chat()


        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        # Accept user input
        if prompt := st.chat_input("What's up?"):
            if uploaded_file is None:
                st.exception(FileNotFoundError("Please upload a document first!"))
                st.stop()

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = bot.ask_question(prompt)
                # context = st.session_state.context

                # Simulate stream of response with milliseconds delay
                # fake_response = "This is a test"
                #for chunk in client.stream(prompt):
                # for chunk in full_response:
                #     full_response += chunk
                #     message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                # st.session_state.context = ctx

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    with col2:
        st.button("Clear ↺", on_click=reset_chat)


if __name__=="__main__":
    main()