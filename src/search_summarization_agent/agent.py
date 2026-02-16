from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from langchain_core.tools import tool
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from deepagents import create_deep_agent


def load_documents(docx_path=None, pdf_path=None):
    documents = []

    if docx_path:
        docx_loader = Docx2txtLoader(docx_path)
        documents.extend(docx_loader.load())

    if pdf_path:
        pdf_loader = PyMuPDFLoader(pdf_path)
        documents.extend(pdf_loader.load())

    return documents


def index_documents(documents, embedding_model):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)

    vector_store = InMemoryVectorStore.from_documents(all_splits, embedding_model)
    return vector_store


def create_agent(vector_store, model):

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    @tool
    def search_documents(query: str) -> str:
        """Search and retrieve information from the uploaded document."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    tools = [search_documents]

    agent = create_deep_agent(
        tools=tools,
        model=model,
    )

    return agent


def main(docx_path=None, pdf_path=None, query=None):

    documents = load_documents(docx_path, pdf_path)

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = index_documents(documents, embedding_model)

    model = ChatOllama(model="mistral-nemo", temperature=0)

    agent = create_agent(vector_store, model)

    response = agent.invoke({"messages": [{"role": "user", "content": query}]})

    final_answer = response["messages"][-1].content
    print(f"Agent Response: {final_answer}")


if __name__ == "__main__":
    # docx_path = "../../src/data/nke-10k-2023.docx"
    docx_path = None
    pdf_path = "../../src/data/nke-10k-2023.pdf"
    query = "What is Nike's principal business activity?"

    main(docx_path, pdf_path, query)
