import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css
from modules import get_simple_conversation, get_pdf_text, get_text_chunks, get_vectorstore_postgres, get_conversation_chain, preprocess_and_predict , handle_user_input





def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple modules :books:")

    # Add combo box for selecting LLM
    # Add combo box for selecting LLM
    llm_model = st.selectbox(
        "Select LLM Model",
        ["llama2", "llama3", "mistral"]
    )
    if llm_model == "llama2":
        model = "codegemma"
    elif llm_model == "llama3":
        model = "llama3"
    elif llm_model == "mistral":
        model = "mistral"
    
    st.write(f"Selected LLM Model: {llm_model}")



    

    

    


    if "chain" not in st.session_state:
        st.session_state.chain = get_simple_conversation(model=model)

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    st.session_state.chain = get_simple_conversation(model=model)

    

    # Container for the conversation
    with st.container(height=500, border=True):

        for message in st.session_state.conversation:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    
    user_question = st.text_input("Ask a question about your documents:", key="fixed_input", label_visibility="hidden", on_change=lambda: handle_user_input())
    with st.sidebar: 
        st.header("settings")
        st.subheader("RAG")
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        embeddings_model = st.selectbox(
        "Select Embeddings Model",
        ["llama2", "llama3", "nomic"]
        )
    
    
        st.write(f"Selected Embeddings Model: {embeddings_model}")
        
        if st.button("Process"):        
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                print("raw_text done")
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                print("text_chunks done")
                # create vector store
                
                vectorstore = get_vectorstore_postgres(text_chunks,embed_model = embeddings_model)
                
                # create conversation chain
                st.session_state.chain = get_conversation_chain(vectorstore)
        
        st.subheader("Predictive Model")


        Prediction_Model = st.selectbox(
        "Select Prediction Model",
        ["DT", "KNN", "MLP", "RF", "SVR"]
        )


        

    st.header("Prediction Model 📊")            
    with st.form(key='prediction_form'):
        
        product_description = st.text_input('Product Description (example: Example Product)')
        collection = st.text_input('Collection (example: Example Collection)')
        market = st.text_input('Market (example: Example Market)')
        channel = st.text_input('Channel (example: Example Channel)')
        subcategory = st.text_input('Subcategory (example: Example Subcategory)')
        technology_code = st.text_input('Technology code (example: Example Technology code)')
        PPHT_1 = st.number_input('PPHT_1 (example: 123.45)', format="%.2f")
        
        submit_button = st.form_submit_button(label='Predict')
    
    if submit_button:
        new_instance = {
            'Product Description': product_description,
            'Collection': collection,       
            'Market': market,
            'Channel': channel,
            'Subcategory': subcategory,
            'Technology code': technology_code,
            'PPHT Y': PPHT_1
            
        }
        
        prediction = preprocess_and_predict(new_instance , Prediction_Model)
        print(prediction)
        st.info(f"Predicted Value of PPHT Y+1: {prediction}")            
                
                
             
    

if __name__ == '__main__':
    main()

# =================================================================================================================================

# def handle_user_input():
#     user_question = st.session_state.fixed_input
#     if user_question:
#         st.session_state.conversation.append({
#             "role": "user",
#             "content": user_question
#         })
        
#         st.session_state.conversation.append({
#             "role": "assistant",
#             "content": st.session_state.chain.run(user_question)
#         })
#         # Clear the input box
#         st.session_state.fixed_input = ""

# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs",
#                        page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)
#     st.header("Chat with multiple modules :books:")

#     # Add combo box for selecting LLM
#     llm_model = st.selectbox(
#         "Select LLM Model",
#         ["llama2", "llama3", "mistral"]
#     )
#     if llm_model == "llama2":
#         model = "codegemma"
#     elif llm_model == "llama3":
#         model = "llama3"
#     elif llm_model == "mistral":
#         model = "mistral"
    
#     st.write(f"Selected LLM Model: {llm_model}")

#     if "chain" not in st.session_state:
#         st.session_state.chain = get_simple_conversation(model=model)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = []

#     st.session_state.chain = get_simple_conversation(model=model)

#     # Container for the conversation
#     with st.container():
#         for message in st.session_state.conversation:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#     # Using st.columns to align text input and button
#     col1, col2 = st.columns([4, 1])  # Adjust the ratio to control width

#     with col1:
#         user_question = st.text_input("Ask a question about your documents:", key="fixed_input", label_visibility="hidden")

#     with col2:
#         # Add some CSS to center align the button vertically
#         button_style = """
#         <style>
#         .stButton > button {
#             height: 2.5em;  /* Adjust height if needed */
#             line-height: 1.5;  /* Adjust line-height if needed */
#             margin-top: 0.5em;  /* Adjust margin to align with text input */
#         }
#         </style>
#         """
#         st.markdown(button_style, unsafe_allow_html=True)
#         if st.button("Send"):
#             handle_user_input()

#     with st.sidebar: 
#         st.header("settings")
#         st.subheader("RAG")
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
#         embeddings_model = st.selectbox(
#         "Select Embeddings Model",
#         ["llama2", "llama3", "nomic"]
#         )
        
#         if embeddings_model == "llama2":
#             embde_model = OllamaEmbeddings(model="llama2", show_progress=True)
#         elif embeddings_model == "nomic":
#             embde_model = NomicEmbeddings(model='nomic-embed-text-v1.5', inference_mode='local', device='gpu')
#         elif embeddings_model == "llama3":
#             embde_model = OllamaEmbeddings(model="llama3", show_progress=True)

#         st.write(f"Selected Embeddings Model: {embeddings_model}")
        
#         if st.button("Process"):        
#             # get pdf text
#             raw_text = get_pdf_text(pdf_docs)
#             print("raw_text done")
#             # get the text chunks
#             text_chunks = get_text_chunks(raw_text)
#             print("text_chunks done")
#             # create vector store
#             vectorstore = get_vectorstore_postgres(text_chunks, embed_model=embeddings_model)
                
#             # create conversation chain
#             st.session_state.chain = get_conversation_chain(vectorstore)
        
#         st.subheader("Predictive Model")

#         Prediction_Model = st.selectbox(
#         "Select Prediction Model",
#         ["DT", "KNN", "MLP", "RF", "SVR"]
#         )

#     st.header("Prediction Model 📊")            
#     with st.form(key='prediction_form'):
        
#         product_description = st.text_input('Product Description (example: Example Product)')
#         collection = st.text_input('Collection (example: Example Collection)')
#         market = st.text_input('Market (example: Example Market)')
#         channel = st.text_input('Channel (example: Example Channel)')
#         subcategory = st.text_input('Subcategory (example: Example Subcategory)')
#         technology_code = st.text_input('Technology code (example: Example Technology code)')
#         PPHT_1 = st.number_input('PPHT_1 (example: 123.45)', format="%.2f")
        
#         submit_button = st.form_submit_button(label='Predict')
    
#     if submit_button:
#         new_instance = {
#             'Product Description': product_description,
#             'Collection': collection,       
#             'Market': market,
#             'Channel': channel,
#             'Subcategory': subcategory,
#             'Technology code': technology_code,
#             'PPHT Y': PPHT_1
#         }
        
#         prediction = preprocess_and_predict(new_instance , Prediction_Model)
#         print(prediction)
#         st.info(f"Predicted Value of PPHT Y+1: {prediction}")

# if __name__ == '__main__':
#     main()
