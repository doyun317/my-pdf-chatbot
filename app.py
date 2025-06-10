# Imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_ollama.llms import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from scripts.document_loader import load_document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import streamlit as st
import numpy as np
from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
import os
import pickle
import hashlib
from pathlib import Path
import re

# Create a Streamlit app
st.title("AI-Powered Document Q&A")

# Create cache directory if it doesn't exist
CACHE_DIR = Path("./embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Load document to streamlit
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# If a file is uploaded, create the TextSplitter and vector database
if uploaded_file:
    # Code to work around document loader from Streamlit and make it readable by langchain
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name

    # Generate cache key based on file content hash
    with open(temp_file, 'rb') as f:
        file_content = f.read()
        cache_key = hashlib.sha256(file_content).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    # Check if cache exists
    if cache_file.exists():
        st.write("Loading cached embeddings... :file_folder:")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            chunks = cached_data['chunks']
            vector_db = cached_data['vector_db']
            bm25 = cached_data['bm25']
    else:
        st.write("Processing document... :watch:")
        # Load document and split it into chunks for efficient retrieval.
        chunks = load_document(temp_file)

        # Generate embeddings using Ollama with nomic-embed-text model
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Create vector database containing chunks and embeddings
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        # Create BM25 index
        tokenized_corpus = [doc.page_content.split() for doc in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        # Save to cache
        cached_data = {
            'chunks': chunks,
            'vector_db': vector_db,
            'bm25': bm25
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)

    # Initialize Cross-Encoder for re-ranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Create a hybrid retriever
    class HybridRetriever(BaseRetriever):
        vector_db: Any = Field(...)
        bm25: Any = Field(...)
        chunks: List[Any] = Field(...)
        cross_encoder: Any = Field(...)
        k: int = Field(default=4)

        def get_relevant_documents(self, query: str) -> List[Any]:
            # Get FAISS results
            faiss_docs = self.vector_db.similarity_search(query, k=self.k)
            
            # Get BM25 results
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_indices = np.argsort(bm25_scores)[-self.k:][::-1]
            bm25_docs = [self.chunks[i] for i in bm25_indices]
            
            # Combine and deduplicate results using page_content as key
            seen = set()
            combined_docs = []
            for doc in faiss_docs + bm25_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    combined_docs.append(doc)
            
            # Re-rank using Cross-Encoder
            if len(combined_docs) > 0:
                # Prepare pairs for re-ranking
                pairs = [(query, doc.page_content) for doc in combined_docs]
                
                # Get re-ranking scores
                scores = self.cross_encoder.predict(pairs)
                
                # Sort documents by re-ranking scores
                doc_score_pairs = list(zip(combined_docs, scores))
                ranked_docs = [doc for doc, score in sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)]
                
                return ranked_docs[:self.k]
            
            return combined_docs[:self.k]

    # Create a hybrid retriever instance
    retriever = HybridRetriever(vector_db=vector_db, bm25=bm25, chunks=chunks, cross_encoder=cross_encoder)
    
    # Initialize Ollama LLM with deepseek-r1:8b model
    llm = OllamaLLM(model="deepseek-r1:8b", device="cuda")

    # Create a system prompt
    system_prompt = (
        "당신은 전문적인 문서 도우미입니다. 주어진 맥락만을 사용하여 사용자의 질문에 답변해야 합니다. "
        "사전 지식을 사용하지 마세요. "
        "중요: 반드시 <think> 태그 안에 사고 과정을 포함해야 합니다. "
        "사고 과정 다음에 태그 없이 최종 답변을 제공하세요. "
        "절대로 영어를 사용하지 마세요. 모든 응답(사고 과정 포함)은 반드시 한글로만 작성해야 합니다. "
        "답변은 명확하게 구조화하고, 필요한 경우 글머리 기호를 사용하세요. "
        "맥락에서 명시적으로 답을 찾을 수 없는 경우, 반드시 '제공된 문서에서는 해당 정보를 찾을 수 없습니다.'라고 답변하세요. "
        "정보를 만들어내지 마세요."
        "{context}"
    )

    # Create a prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create a chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Creates the RAG
    chain = create_retrieval_chain(retriever, question_answer_chain)

    # Streamlit input for question
    question = st.text_input("Ask a question about the document:")
    if question:
        # Answer
        response = chain.invoke({"input": question})['answer']
        
        # Extract think content and final answer
        think_pattern = r'<think>(.*?)</think>'
        think_content = re.search(think_pattern, response, re.DOTALL)
        final_answer = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
        
        # Display think content in collapsible section if it exists
        if think_content:
            with st.expander("🤔 사고 과정 보기", expanded=False):
                st.markdown(think_content.group(1).strip())
        
        # Display final answer
        st.markdown("### 답변")
        st.markdown(final_answer)
    
