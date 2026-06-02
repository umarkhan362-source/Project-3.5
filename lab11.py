import os
import glob
import numpy as np
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from google import genai

# 1. Environment variables load karna (.env se GEMINI_API_KEY parhnay k liye)
load_dotenv()

# ==========================================================
# STEP 1: INITIALIZE CHROMADB WITH LOCAL MODEL
# ==========================================================
print("🚀 Step 1: Initializing Local ChromaDB with SentenceTransformers...")
chroma_client = chromadb.PersistentClient(path='./chroma_db')

local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2'
)

collection = chroma_client.get_or_create_collection(
    name='company_docs',
    embedding_function=local_ef,
    metadata={'description': 'Company policy documents'}
)
print(f"📦 ChromaDB Collection Status: Ready (Current Count: {collection.count()})")

# ==========================================================
# STEP 2: PLAIN PYTHON INGESTION & TEXT SPLITTING
# ==========================================================
print("\n📂 Step 2: Processing Ingestion Pipeline (Plain Python Style)...")

if not os.path.exists('company_docs'):
    os.makedirs('company_docs')
    with open('company_docs/sample_policy.txt', 'w', encoding='utf-8') as f:
        f.write("Vacation Policy: Employees get 20 days of paid time off (PTO) annually. "
                "Dress Code: Casual attire is allowed from Monday to Thursday, Fridays are formal.")
    print("💡 Target directory created with a default 'sample_policy.txt' file.")

chunks = []
txt_files = glob.glob('company_docs/*.txt')

for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunk_size = 500
    overlap = 50
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end]
        chunks.append({"text": chunk_text, "source": os.path.basename(file_path)})
        start += (chunk_size - overlap)

if collection.count() == 0 and chunks:
    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[f'doc_{i}' for i in range(len(chunks))],
        metadatas=[{'source': c["source"]} for c in chunks]
    )
    print(f'✅ Sync Complete: Ingested {len(chunks)} text chunks into Vector DB.')
else:
    print(f'ℹ️ DB Sync Skipped: Vector storage already has {collection.count()} indexed states.')


# ==========================================================
# STEP 3: MANUALLY CODED SEARCH FUNCTIONS
# ==========================================================
def keyword_search(query, chunks_list, top_k=2):
    query_lower = query.lower()
    scored = []
    for chunk in chunks_list:
        score = sum(chunk["text"].lower().count(word) for word in query_lower.split())
        if score > 0:
            scored.append((score, chunk["text"]))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [text for score, text in scored[:top_k]]


def vector_search(query, n_results=2):
    return collection.query(query_texts=[query], n_results=n_results)


# ==========================================================
# STEP 4: GENERATIVE RAG WITH FIXED GEMINI MAPPING
# ==========================================================
print("\n🧠 Step 3: Binding RAG Interface with Native Google GenAI Client...")
google_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))


def semantic_rag(query, n_results=2):
    """Complete End-to-End Semantic RAG Pipeline with fixed text generation model mapping."""
    results = vector_search(query, n_results=n_results)

    if not results['documents'] or not results['documents'][0]:
        return 'No relevant database records matching semantic space.'

    context = '\n\n---\n\n'.join(results['documents'][0])

    prompt = f'''You are a helpful HR assistant. Answer using ONLY the context provided below. 
If the query cannot be answered using the explicit context fields, say "I cannot find this in the policy documents."

Context:
{context}

Question: {query}
Answer:'''

    try:
        # FIXED MODEL PARAMETER FOR THE NEW SDK COMPATIBILITY
        response = google_client.models.generate_content(
            model='gemini-2.5-flash',  # <-- Naye standard package ka production model string
            contents=prompt
        )
        return response.text
    except Exception as e:
        # Fallback agar backend server dynamic routing block kare
        return f"Generation Error: {e}"


# ==========================================================
# EXECUTION BENCHMARKS & EXPERIMENTS
# ==========================================================
print('\n🔍 === TASK 1: RUNNING SYNTACTIC VS SEMANTIC COMPARISON ===')
comparison_query = 'PTO policy'

print('\n❌ TRADITIONAL KEYWORD SEARCH (Week 10):')
kw_res = keyword_search(comparison_query, chunks, top_k=2)
print(f'Found: {len(kw_res)} text context matches.')
for idx, text_block in enumerate(kw_res):
    print(f'  [{idx + 1}] {text_block[:120]}...')

print('\n🧠 CHROMADB SEMANTIC VECTOR SEARCH (Week 11):')
sem_res = vector_search(comparison_query, n_results=2)
print(f'Found: {len(sem_res["documents"][0])} semantic matches.')
if sem_res["documents"][0]:
    print(f'  [Top Match]: {sem_res["documents"][0][0][:120]}...')

print('\n🏁 === TASK 2: LIVE END-TO-END RAG PIPELINE DEMO ===')
rag_scenarios = [
    'How much time off do employees get?',
    'Can I work from home?',
    'What is the dress code for Friday?'
]

for q in rag_scenarios:
    print(f'\n❓ User Query: "{q}"')
    print(f'🤖 RAG Response: {semantic_rag(q)}')
