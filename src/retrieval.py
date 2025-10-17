import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.schema import get_schema

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# validar variables mínimas
if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD and OPENAI_API_KEY):
    raise SystemExit("Faltan variables de entorno. Copia .env.example -> .env y rellena las claves.")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

llm = OpenAILLM(model_name=OPENAI_MODEL, api_key=OPENAI_API_KEY)
embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Initialize the retriever
retriever = VectorRetriever(
    driver,
    index_name="chunkEmbeddings",
    embedder=embedder,
    return_properties=["text"]
)
query = "Gabriela dos Reis Bueno? fala algum idioma?"
rag = GraphRAG(llm=llm, retriever=retriever)
print(rag.search(query).answer)