import os
import sys
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate 
from langchain.chains import ConversationalRetrievalChain 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.memory import ConversationBufferWindowMemory 
from streamlit.components.v1 import html

# ----- Configuration and Setup -----
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
    sys.exit(1)

# ----- Initialize Gemini Model -----
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.3 
)

# --- Embedding Model Setup ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# --- Connect to Chroma Database ---
vectordb = Chroma(
    persist_directory="./career_chroma_db", # Path to stored embeddings
    embedding_function=embeddings
    )


# ----- RAG & AI Functions -----

# Prepare prompt to send to Gemini
qa_prompt = PromptTemplate.from_template("""
Sen bir kariyer danışmanısın.
Kullanıcının sorusunu, aşağıdaki bağlam (context) parçalarından yararlanarak ayrıntılı ve anlaşılır şekilde yanıtla.
Sadece bağlamda bulunan bilgileri kullan ama aynı alan içindeki benzer kavramları da çeşitlendir.

Kullanıcı sorusu:
{question}

Bağlam (veri setinden alıntılar):
{context}

- Eğer sohbet geçmişi yoksa: detaylı, madde madde açıklayıcı yanıt ver.
  * Uygun meslek alanları (en fazla 5)
  * Hangi üniversite bölümleri için uygundur (Field of Study)
  * Bu alanlarda öne çıkan beceriler (3–6 madde)
  * Kısa hazırlık planı (madde madde, 4–5 adım)
            
- Eğer sohbet geçmişi varsa: tekrara düşmeden, kısa ve odaklı cevap ver.
- Gerektiğinde benzer meslekleri de grupla ama mümkünse farklı alt dallardan örnekler ver.
""")

# Get similar documents from vector database
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.1,
        "k": 5
    }
)

# --- Memory Setup ---
memory = ConversationBufferWindowMemory(
        k=3,
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer"
)

# --- Combine into Conversational RAG Chain ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=False
)

# --- STREAMLIT UI Setup ---
st.set_page_config(page_title="Career Mentor Chatbot", page_icon="💬")

# CSS
with open("static/style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# HEADER
with open("templates/header.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display welcome message only when no messages are present
if not st.session_state.messages:
    with open("templates/welcome_message.html", "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)


# Display  chat messages
for msg in st.session_state.messages:
    role = msg["role"]
    text = msg["content"]
    bubble_class = "user-bubble" if role == "user" else "ai-bubble"

    if role == "user":
        # Render user message
        st.markdown(f"<div class='{bubble_class}'>{text}</div>", unsafe_allow_html=True)
    else:
       # AI response rendering logic
        st.markdown(
            f"<div class='{bubble_class}'>",
            unsafe_allow_html=True
        )
        st.markdown(text, unsafe_allow_html=False)
        st.markdown("</div>", unsafe_allow_html=True)

user_input = st.chat_input("Mesajını yaz...")

if user_input:
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate AI response
    with st.spinner("Yanıt hazırlanıyor..."):
        # Invoke the QA chain to fetch the AI response
        result = qa_chain.invoke({"question": user_input})
        ai_response = result["answer"] if "answer" in result else result.get("result", "")

    st.session_state.messages.append({"role": "ai", "content": ai_response})
    st.rerun()

