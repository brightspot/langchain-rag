import json
import os
import requests
import sys
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

# Simple retriever that fetches from Brightspot GraphQL endpoint.
class BrightspotRetriever(BaseRetriever):

    endpoint_url: str

    def __init__(self, endpoint_url: str):
        super().__init__(endpoint_url=endpoint_url)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        data = {"query": query}
        response = requests.post(self.endpoint_url, data=data)

        response.raise_for_status()

        items = response.json()
        documents = []

        for item in items:
            documents.append(Document(page_content=json.dumps(item)))

        return documents

def rag(llm, retriever, question):
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

    print()
    print("Question:")
    print(question)
    print()

    result = chain.invoke({"input": question})

    print("Answer:")
    print(result["answer"])
    print()

if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print(f"Usage: {sys.argv[0]} '<openai api key>' '<brightspot endpoint url>' '<question>'")

    else:
        openai_api_key, brightspot_endpoint_url, question = sys.argv[1:]
        llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        retriever = BrightspotRetriever(endpoint_url=brightspot_endpoint_url)

        rag(llm, retriever, question)
