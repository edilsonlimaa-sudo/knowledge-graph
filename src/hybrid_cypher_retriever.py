import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
# Importando o componente HybridCypherRetriever
from neo4j_graphrag.retrievers import HybridCypherRetriever 
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.schema import get_schema

# --- Configurações ---
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Nomes de índices (ASSUMIDOS: Mantenha estes nomes consistentes no Neo4j)
VECTOR_INDEX_NAME = "chunkEmbeddings" # Para busca vetorial (similaridade)
FULLTEXT_INDEX_NAME = "candidateFulltext" # Para busca lexical

if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD and OPENAI_API_KEY):
    raise SystemExit("Faltam variáveis de entorno.")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

llm = OpenAILLM(model_name=OPENAI_MODEL, api_key=OPENAI_API_KEY)
embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# --- QUERY DE RETRIEVAL CYPER BASEADA NO SEU ESQUEMA ---
# Objetivo: Encontrar o nó de Pessoa mais próximo e fazer o "Traversal" 
# para pegar TODA a experiência de trabalho e competências.
RETRIEVAL_QUERY = f"""
    // 1. Encontra a Pessoa associada ao Chunk encontrado
    // USANDO A RELAÇÃO REAL DO SEU GRAFO: :FROM_CHUNK
    MATCH (node)-[:FROM_CHUNK*0..2]-(p:Pessoa)

    // 2. Coleta as experiências de trabalho (TRABALHOU_EM)
    OPTIONAL MATCH (p)-[r_trab:TRABALHOU_EM]->(o:Organização)
    
    // 3. Coleta as competências (TEM_COMPETENCIA)
    OPTIONAL MATCH (p)-[r_comp:TEM_COMPETENCIA]->(c:Competencia)
    
    // Agrupa e retorna o contexto rico
    RETURN
        p.nome AS candidateName,
        // p.email foi removido, pois o campo não está consistentemente populado.
        
        collect(DISTINCT {{
            organização: o.nome,
            entrada: r_trab.data_entrada, 
            saida: r_trab.data_saida
        }}) AS experienciaTrabalho,
        
        collect(DISTINCT {{
            competencia: c.nome,
            // nivel_proficiencia foi removido, pois o campo não está consistentemente populado.
            anos: r_comp.anos_experiencia
        }}) AS competencias,
        
        node.text AS chunkContext, // Mantém o chunk original como contexto semântico
        score AS similarityScore
"""

# --- Inicialização do HybridCypherRetriever ---
retriever = HybridCypherRetriever(
    driver=driver,
    vector_index_name=VECTOR_INDEX_NAME,
    fulltext_index_name=FULLTEXT_INDEX_NAME,
    embedder=embedder,
    retrieval_query=RETRIEVAL_QUERY,
)

query = "Dos candidatos que temos na base quais podem estar mais alinhados para liderar a arquitetura de um projeto?"
rag = GraphRAG(llm=llm, retriever=retriever)

print("\n--- Resultado do HybridCypherRetriever Otimizado ---")
print(f"Buscando por: {query}\n")
print(rag.search(query).answer)

# Fechar a conexão
driver.close()