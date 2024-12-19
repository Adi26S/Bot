from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import PineconeException


embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
load_dotenv()


def ingest_docs():
    loader = ReadTheDocsLoader("/Users/apple/Desktop/documentation-helper/langchain-docs/api.python.langchain.com/en/latest")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata['source']
        new_url = new_url.replace("lanchain-docs","https:/")
        doc.metadata.update({"source":new_url})

    print(f"Going to add {len(documents)} to pincone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name = "langchain-docs-index")

    print("**** Loading to vectorstore done*****")





if __name__=="__main__":
    ingest_docs()
