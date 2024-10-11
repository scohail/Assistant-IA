from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
import pandas as pd
import joblib
import torch
from langchain.vectorstores.pgvector import PGVector
from sqlalchemy import create_engine
from langchain_nomic   import NomicEmbeddings
from sqlalchemy import create_engine, text
import uuid
import json

# La lecture des pdfs  et l'extraction du texte



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


def get_vectorstore(text_chunks ):
    print("getting vectorstore")
    embeddings = OllamaEmbeddings(model="llama2", show_progress=True)
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    #save the vectorstore
    vectorstore.save("vectorstore")

    return vectorstore


# def get_vectorstore_postgres(text_chunks,embed_model = NomicEmbeddings(model='nomic-embed-text-v1.5',inference_mode='local',device='gpu' )):
    print("getting vectorstore")
   
    

    # PostgreSQL connection details
    db_url = "postgresql+psycopg2://postgres:root@database.datatika.online:5431/vectordb"
    
    # Create SQLAlchemy engine
    # engine = create_engine(db_url)  

    

    
    # Create PGVector instance


    if embed_model == "llama2":
        pg_vector= PGVector.from_texts(texts=text_chunks, connection_string=db_url  ,embedding= OllamaEmbeddings(model="llama2", show_progress=True),collection_name="llama_embeddings" ) 

    vectorstore = PGVector.from_texts(texts=text_chunks, connection_string=db_url  ,embedding= embed_model  )   


    return vectorstore  
                

def create_table_if_not_exists(connection, table_name):

    print("creatinG QUERY")

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        collection_id TEXT NOT NULL,
        embedding VECTOR(4096),
        document TEXT NOT NULL,
        cmetadata JSONB,
        custom_id TEXT,
        uuid UUID DEFAULT gen_random_uuid()
    );
    """
    print("creating table")
    connection.execute(text(create_table_query))
    connection.commit()  # Valider la création de la table
    print(f"Table {table_name} créée ou existante")



# Fonction pour insérer les embeddings en utilisant PGVector
def get_vectorstore_postgres(text_chunks, embed_model):
    print("getting vectorstore")

    # PostgreSQL connection details
    db_url = "postgresql+psycopg2://postgres:root@database.datatika.online:5431/vectordb"
    
    # Créer l'instance du moteur SQLAlchemy
    print("creating engine")
    engine = create_engine(db_url)
    print("engine created") 

    print("embed_model: ", embed_model)
    # Définir le modèle et la taille du vecteur
    if embed_model == "llama2":
        embed_model_instance = OllamaEmbeddings(model="llama2", show_progress=True)
        collection_name = "llama2_embeddings"         
        vector_size = 4096  # Adapter selon le modèle
    elif embed_model == "nomic":
        embed_model_instance = NomicEmbeddings(model='nomic-embed-text-v1.5', inference_mode='local', device='gpu')
        collection_name = "nomic_embeddings"
        vector_size = 768  # Adapter selon le modèle
    elif embed_model == "llama3":
        embed_model_instance = OllamaEmbeddings(model="llama3", show_progress=True)
        collection_name = "llama3_embeddings"
        vector_size = 4096
    
    else:
        raise ValueError("Embedder non supporté.")
    print("embed_model_instance: ", embed_model_instance)
    # Vérifier et créer la table si nécessaire

    print("creating table if not exists")   
    with engine.connect() as connection:
        create_table_if_not_exists(connection, collection_name)
    print(f"text chunks1: {text_chunks}")
    # Utiliser PGVector pour générer les embeddings
    embeddings = embed_model_instance.embed_documents(text_chunks)

    # Insérer les embeddings avec leurs métadonnées

    print(f"text chunks2: {text_chunks}")

    with engine.connect() as connection:
        for i, (text_chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            insert_query = f"""
            INSERT INTO {collection_name} (collection_id, embedding, document, cmetadata, custom_id, uuid)
            VALUES (:collection_id, :embedding, :document, :cmetadata, :custom_id, :uuid)
            """
            connection.execute(text(insert_query), {
                'collection_id': collection_name,
                'embedding': embedding,
                'document': text_chunk,
                'cmetadata': json.dumps({}),  # Convertir en chaîne JSON
                'custom_id': f"doc_{i}",
                'uuid': uuid.uuid4()  # Génération d'un UUID
            })
        connection.commit()  # Valider les insertions

    print(f"Embeddings insérés dans la table {collection_name}")

    pgvector=PGVector(
       
        connection_string=db_url,
        embedding_function=embed_model_instance,
        collection_name="nomic_embeddings"
    )
    print(f'pgvector: {pgvector}')
    return pgvector



def get_conversation_chain(vectorstore, model="llama3"):
    print("getting conversation chain")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=Ollama(model=model),
        retriever=vectorstore.as_retriever(),
        memory=memory, 
    )
    
    
    print("conversation_chain done")
    

    return conversation_chain



def get_simple_conversation(model = "llama3"):
    template_messages = [
        SystemMessage(content="You are a helpfull assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(template_messages)
        
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = LLMChain(llm = Ollama(model=model), prompt=prompt_template, memory=memory)
 
    return chain




def preprocess_and_predict( single_instance, model = "RF"):
    # Convert new_instance to DataFrame
    single_instance = pd.DataFrame([single_instance])
    
    # Apply the same transformations as training 

    reference_path = '/home/scohail/Desktop/RAG/ask-multiple-pdfs/Saved_Models/(here)_model_1.pkl'

    model_path = reference_path.replace('(here)', model)

    scaler = joblib.load('/home/scohail/Desktop/RAG/ask-multiple-pdfs/Saved_Models/scaler_1.pkl')
    pca = joblib.load('/home/scohail/Desktop/RAG/ask-multiple-pdfs/Saved_Models/pca_1.pkl')
    product_encoder = joblib.load('/home/scohail/Desktop/RAG/ask-multiple-pdfs/Saved_Models/product_encoder.pkl')
    collection_encoder = joblib.load('/home/scohail/Desktop/RAG/ask-multiple-pdfs/Saved_Models/collection_encoder.pkl')
    model = joblib.load(model_path)
    columns = joblib.load('/home/scohail/Desktop/RAG/ask-multiple-pdfs/Saved_Models/columns_1.pkl')



    single_instance['Product Description'] = product_encoder.transform(single_instance['Product Description'])
    single_instance['Collection'] = collection_encoder.transform(single_instance['Collection'])
    single_instance = pd.get_dummies(single_instance, columns=['Market', 'Channel', 'Subcategory', 'Technology code'])
    
    # Ensure the new instance has the same columns as the training data
    single_instance = single_instance.reindex(columns=columns, fill_value=0)
    
    # Perform the same scaling and PCA transformations
    single_instance_scaled = scaler.transform(single_instance)
    single_instance_pca = pca.transform(single_instance_scaled)
    
    # Make prediction
    prediction = model.predict(single_instance_pca)
    return prediction


def handle_user_input():
    user_question = st.session_state.fixed_input
    if user_question:
        st.session_state.conversation.append({
            "role": "user",
            "content": user_question
        })
        
        st.session_state.conversation.append({
            "role": "assistant",
            "content": st.session_state.chain.run(user_question)
        })
        # Clear the input box
        st.session_state.fixed_input = ""
