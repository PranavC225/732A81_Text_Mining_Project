import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

CHROMA_DIR  = "data/chroma"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

PROMPT_TEMPLATE = """You are a helpful assistant for international students at Linköping University.
Answer the question using only the context provided below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

def load_chain(top_k: int = 8):
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        collection_name="liu_housing",
        embedding_function=embedder,
        persist_directory=CHROMA_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = Ollama(base_url=OLLAMA_HOST, model="llama3.2:3b", temperature=0.1)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    print("Loading chain...")
    chain = load_chain()
    print("LiU Housing Q&A Agent ready. Type 'quit' to exit.\n")
    while True:
        q = input("Question: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        print("\nAnswer:", chain.invoke(q), "\n")