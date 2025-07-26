import os, textwrap
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("No API KEY Found")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

PDF_DIR = Path("data/pdfs")  
pdf_paths = list(PDF_DIR.glob("*.pdf"))
if not pdf_paths:
    raise FileNotFoundError("No PDFs found in ./data/pdfs/")

raw_docs: List[Dict] = []
for path in pdf_paths:
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                raw_docs.append({
                    "content": text,
                    "source": path.name,
                    "page": page_num,
                })
print(f"Loaded {len(raw_docs)} pages from {len(pdf_paths)} PDFs")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
    length_function=len,
)
chunks = []
for doc in raw_docs:
    for chunk in splitter.split_text(doc["content"]):
        chunks.append({
            "content": chunk,
            "source": doc["source"],
            "page": doc["page"],
        })
print(f"Total chunks: {len(chunks)}")

emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(
    texts=[c["content"] for c in chunks],
    embedding=emb_model,
    metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
)
# Persist index
vector_db.save_local("faiss_index")
print("FAISS index saved to ./faiss_index")


MAX_CHUNKS = 5  # top‑k to include in prompt

def ask_gemini(query: str):
    # 1. Retrieve top relevant chunks
    docs = vector_db.similarity_search(query, k=MAX_CHUNKS)

    # 2. Build context blocks and collect references
    context_blocks = []
    references = set()
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        context_blocks.append(
            f"[doc{i} | {meta['source']} | page {meta['page']}]\n{d.page_content}"
        )
        references.add((meta['source'], meta['page']))

    # 3. Build prompt
    context_text = "\n\n".join(context_blocks)
    prompt = textwrap.dedent(f"""
        You are an expert integration architect. Use the following source excerpts to answer
        the question. If the question is not related to the pdfs then say "I don't know".
        Keep answer under 250 words.

        <context>\n{context_text}\n</context>

        USER QUESTION: {query}
    """)

    # 4. Generate answer
    response = gemini_model.generate_content(prompt)
    answer = response.text.strip()

    # 5. Format references
    ref_list = sorted(references)
    citations = "; ".join([f"{src} (Page {pg})" for src, pg in ref_list])

    if answer.lower().startswith("i don't know."):
        citations = ''
    return answer, citations


if __name__ == "__main__":
    question = "What will happen if Corner 5 is down?"
    answer, sources = ask_gemini(question)
    print("Answer:\n", answer)
    print("\nSources:\n", sources)