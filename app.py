import os
import chromadb
from chromadb.utils import embedding_functions
import requests
import streamlit as st
from dotenv import load_dotenv

# ==================== FORCE DISABLE VERTEX AI ====================
if "GOOGLE_GENAI_USE_VERTEXAI" in os.environ:
    del os.environ["GOOGLE_GENAI_USE_VERTEXAI"]

os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)   # Extra safety

# ==========================================================
# 🛠️ TASK 3.3: GLOBAL ERROR HANDLING (Start of Try Block)
# ==========================================================
try:
    # Background memory se environment variables load karna
    load_dotenv()

    # ==========================================================
    # ⚡ CACHED INITIALIZATION (Runs once for efficiency)
    # ==========================================================
    @st.cache_resource
    def init_chromadb():
        """
        Initialize ChromaDB with EXACT Lab 11 settings
        """
        client = chromadb.PersistentClient(path='./chroma_db')
        
        # Lab 11 wala exact stable embedding model
        local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        
        # Agar folder missing hoga toh automatic FileNotFoundError catch hoga
        collection = client.get_collection(
            name='company_docs',
            embedding_function=local_ef
        )
        return collection

    # Global Variable for DB
    collection = init_chromadb()

    # Streamlit secrets se key nikal kar clean karna
    if "GEMINI_API_KEY" in st.secrets:
        raw_api_key = st.secrets["GEMINI_API_KEY"]
    else:
        raw_api_key = os.getenv("GEMINI_API_KEY")
         
    if not raw_api_key:
        raise ValueError("Missing Gemini API Key")
         
    clean_api_key = raw_api_key.strip().replace('"', '').replace("'", "")

    # CORRECTED INDENTATION: Function is now safely aligned inside the try block
    def get_rag_response(query, n_results=3):
        """
        Get answer using RAG with Gemini API (Fixed Model Name)
        """
        try:
            # Search vector database
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results.get('documents') or not results['documents'][0]:
                return 'No relevant information found in documents.'
            
            context = '\n\n--\n\n'.join(results['documents'][0])
            
            prompt = f'''You are a helpful HR assistant. Answer using ONLY the context below. 
If the answer is not in the context, say "I cannot find this information in the company policy documents."
Be concise and professional.

Context:
{context}

Question: {query}
Answer:'''

            # FIXED: Use versioned model name
            url = url =url = f"https://generativelanguage.googleapis.com/v1/models/gemini-3.5-flash:generateContent?key={clean_api_key}"
            
            headers = {'Content-Type': 'application/json'}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 800,
                }
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response_data = response.json()
            
            if response.status_code == 200:
                return response_data['candidates'][0]['content']['parts'][0]['text']
            else:
                error_msg = response_data.get('error', {}).get('message', response.text)
                return f"❌ API Error ({response.status_code}): {error_msg}"
                
        except Exception as e:
            return f'Error: {str(e)}'

    # ==========================================================
    # 🎨 STREAMLIT MAIN UI DESIGN
    # ==========================================================
    st.set_page_config(
        page_title='AI Assistant',
        page_icon='🤖',
        layout='wide'
    )

    st.title('🤖 Company Knowledge Assistant')
    st.markdown('Ask me anything about company policies, vacation rules, or dress codes!')

    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # ==========================================================
    # 🛠️ TASK 3.2: ADD WELCOME MESSAGE
    # ==========================================================
    if len(st.session_state.messages) == 0:
        welcome = '''
        Hi! I'm your company knowledge assistant.
        
        I can help you find information about:
        - Vacation and time off policies
        - Remote work guidelines
        - Parental leave benefits
        - And more!
        
        Just ask me a question to get started.
        '''
        with st.chat_message('assistant'):
            st.write(welcome)

    # ==========================================================
    # 🛠️ TASK 3.1: ADD SIDEBAR
    # ==========================================================
    with st.sidebar:
        st.header('About')
        st.markdown('''
        This AI assistant can answer questions about:
        - Vacation policies
        - Remote work guidelines
        - Parental leave
        - Benefits information
         
        Powered by:
        - Gemini 1.5 Flash (v1 Production)
        - ChromaDB vector search
        - Semantic RAG
        ''')
         
        st.divider()
         
        # Stats (Live metrics display)
        st.metric('Documents Indexed', collection.count() if collection else 0)
        st.metric('Messages in Chat', len(st.session_state.messages))
         
        st.divider()
         
        # Clear chat button
        if st.button('Clear Chat History'):
            st.session_state.messages = []
            st.rerun()

    # ==========================================================
    # 💬 CHAT INTERFACE LOGIC
    # ==========================================================

    # Display chat history from session state
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

    # Chat input section
    if prompt := st.chat_input('Ask a question...'):
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.write(prompt)
              
        # Get AI response
        with st.chat_message('assistant'):
            with st.spinner('Searching documents...'):
                response = get_rag_response(prompt)
            st.write(response)
              
        # Save response
        st.session_state.messages.append({'role': 'assistant', 'content': response})

# ==========================================================
# 🛠️ TASK 3.3: EXCEPT BLOCKS (Catching Global Errors)
# ==========================================================
except FileNotFoundError:
    st.error('''
    Error: ChromaDB not found.
    
    Please run Week 11 lab to create the vector database first.
    ''')
    st.stop()
    
except Exception as e:
    st.error(f'Error: {str(e)}')
    st.info('Make sure your Streamlit Secrets has GEMINI_API_KEY set.')
    st.stop()
