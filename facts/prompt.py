from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# Functionality testing
# result = embeddings.embed_query(# "hi there")
# print(result)

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings,  # just happens to be different than Chroma.from_documents
)

# another test, for some reeason it returns an empty array
# result = db.similarity_search("what is an interesting fact about people")
# print("result", result)

# third test with already stored vector
emb = embeddings.embed_query("what is an interesting fact about...")

# Find similar documents by using the embedding we just calculated
results = db.similarity_search_by_vector(emb)
print(results)

# retriever = db.as_retriever()
retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
    # chain_type="refine",
    # chain_type="map_redice",
    verbose=True,
    # chain_type="map_rerank",
)  # chain type stuff is basic, stuff it into prompt


result = chain.run("what is an interesting fact about the English language")

print(result)
