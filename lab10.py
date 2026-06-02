import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# STEP 1: INITIALIZATION & ENVIRONMENT SETUP
# ==========================================

# Paste your FRESH Gemini API key directly here inside the quotes
NEW_GEMINI_KEY = "AIzaSyDVwcTGaH21jrW8zwTQP1YYFSfqKdrOsdA"

# Pass it explicitly to clear the Pydantic validation error instantly
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=NEW_GEMINI_KEY,
    temperature=0.7
)

print("Gemini LLM initialized successfully with direct API key!")
# ==========================================
# STEP 2: LOAD DOCUMENTS
# ==========================================
loader = DirectoryLoader(
    'company_docs/',
    glob='*.txt',
    loader_cls=TextLoader
)
documents = loader.load()

# ==========================================
# STEP 3: CHUNK DOCUMENTS
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=['\n\n', '\n', '. ', ' ', '']
)
chunks = text_splitter.split_documents(documents)


# ==========================================
# STEP 4: RETRIEVAL ALGORITHM (Your Search Function)
# ==========================================
def simple_search(query, chunks, top_k=3):
    """
    Simple keyword-based search
    Returns top_k most relevant chunks
    """
    query_lower = query.lower()
    scored_chunks = []

    for chunk in chunks:
        content_lower = chunk.page_content.lower()
        score = 0
        for word in query_lower.split():
            score += content_lower.count(word)

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:top_k]]


# ==========================================
# STEP 5: EXECUTE AND TEST
# ==========================================
query = 'What is the vacation policy?'
relevant = simple_search(query, chunks)

print(f'Found {len(relevant)} relevant chunks:')
for i, chunk in enumerate(relevant):
    print(f'\n--- Chunk {i + 1} ---')
    print(chunk.page_content)

# ==========================================
# STEP 6: TEST MULTIPLE QUERIES
# ==========================================
print("\n" + "=" * 40)
print("RUNNING MULTIPLE POLICY TESTS...")
print("=" * 40)

test_queries = [
    'How many vacation days do employees get?',
    'What is the remote work policy?',
    'Tell me about parental leave',
]

for query in test_queries:
    print(f'\nQuery: "{query}"')

    # Run your keyword matching algorithm across all chunks, grabbing top 2 matches
    results = simple_search(query, chunks, top_k=2)

    print(f'-> Found {len(results)} relevant chunks')

    if results:
        # Strip any extra newlines from the snippet for a cleaner terminal printout
        clean_snippet = results[0].page_content.replace('\n', ' ').strip()
        print(f'   Top result preview: {clean_snippet[:120]}...')


# ==========================================
# STEP 7: THE RAG PIPELINE (Retrieve → Generate)
# ==========================================

def rag_query(query, chunks, top_k=3):
    """
    RAG pipeline: Retrieve relevant chunks → Construct bounded prompt → Generate precise answer
    """
    # 1. Retrieve the most relevant text chunks using your keyword search function
    relevant_chunks = simple_search(query, chunks, top_k)

    if not relevant_chunks:
        return 'No relevant information found in documents.'

    # 2. Build cohesive context by joining the matched chunks together
    context = '\n\n---\n\n'.join([
        chunk.page_content for chunk in relevant_chunks
    ])

    # 3. Construct a strict, grounded prompt template forcing the LLM to stay factual
    prompt = f'''You are a helpful assistant. Answer the question using ONLY the context provided below. 
If the answer is not in the context, say so. Do not invent facts.

Context:
{context}

Question: {query}
Answer:'''

    # 4. Generate the final answer using your pre-initialized Gemini LLM
    messages = [{'role': 'user', 'content': prompt}]
    response = llm.invoke(messages)

    return response.content


# ==========================================
# STEP 8: TEST THE COMPLETE RAG SYSTEM
# ==========================================
print("\n" + "=" * 40)
print("TESTING FULL RAG SYSTEM RESPONSES")
print("=" * 40)

# Test 1: Asking an explicitly supported policy question
rag_answer = rag_query("What is the remote work policy?", chunks)
print("\nUser Query: 'What is the remote work policy?'")
print(f"RAG Bot Answer:\n{rag_answer}")

# Test 2: Asking a question completely absent from your documents to test constraints
out_of_bounds_answer = rag_query("What is the company dress code policy?", chunks)
print("\nUser Query: 'What is the company dress code policy?'")
print(f"RAG Bot Answer:\n{out_of_bounds_answer}")


import time  # Import Python's built-in time module

''''# ==========================================
# STEP 8: FINAL RAG BATCH TESTING (WITH RATE LIMIT PROTECTION)
# ==========================================
questions = [
     'How many vacation days do full-time employees get?',
     'Can employees work from home?',
     'What is the parental leave policy?',
     'What is the dress code?'
]

for question in questions:
     print(f'\n{"="*60}')
     print(f'Q: {question}')
     print(f'{"="*60}')

     try:
         # Run the query through your pipeline
         answer = rag_query(question, chunks)
         print(f'A: {answer}')

         # Crucial: Pause the script for 4 seconds to avoid hitting the RPM limit
         print("...Pausing 4 seconds to respect API rate limits...")

         time.sleep(4)

     except Exception as e:
         print(f"⚠️ API Error encountered: {e}")
         print("If you hit a 429 quota limit, wait a few minutes or use an alternate API project key.")'''
# ==========================================
# STEP 9: RAG VS. DIRECT LLM COMPARISON
# ==========================================
import time

def ask_without_rag(question):
     """
     Ask the LLM directly without passing any custom document context
     """
     messages = [
          {'role': 'system', 'content': 'You are a helpful HR assistant.'},
          {'role': 'user', 'content': question}
     ]
     response = llm.invoke(messages)
     return response.content

# Run the benchmark test
question = 'How many vacation days do employees get?'

print('\n' + '='*50)
print('BENCHMARK: DIRECT LLM VS. RAG SYSTEM')
print('='*50)

# 1. Test without context
print('\n[1] WITHOUT RAG (Direct LLM Guess):')
try:
    direct_answer = ask_without_rag(question)
    print(direct_answer)
except Exception as e:
    print(f"Direct API Error: {e}")

# Pause to protect your free daily/per-minute request quota
print("\n...Pausing 4 seconds to protect API limits...")
time.sleep(4)

# 2. Test with your custom RAG context
print('\n[2] WITH RAG (Grounded in Company Docs):')
try:
    rag_answer = rag_query(question, chunks)
    print(rag_answer)
except Exception as e:
    print(f"RAG API Error: {e}")