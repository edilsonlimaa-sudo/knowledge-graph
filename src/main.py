from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.indexes import create_vector_index
import nest_asyncio
import asyncio
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD and OPENAI_API_KEY):
    raise SystemExit("Faltan variables de entorno. Copia .env.example -> .env y rellena las claves.")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
llm = OpenAILLM(model_name=OPENAI_MODEL, api_key=OPENAI_API_KEY)
embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# ------------------ Node Types ------------------
NODE_TYPES = [
    {"label": "Pessoa", "properties": [{"name": "nome", "type": "STRING"}, {"name": "email", "type": "STRING"}]},
    {"label": "Profissional", "properties": [{"name": "senioridade", "type": "STRING"}, {"name": "cargo_atual", "type": "STRING"}]},
    {"label": "Organização", "properties": [{"name": "nome", "type": "STRING"}, {"name": "setor", "type": "STRING"}]},
    {"label": "Empresa", "properties": []},
    {"label": "Cliente", "properties": []},
    {"label": "Competencia", "properties": [{"name": "nome", "type": "STRING"}, {"name": "tipo_macro", "type": "STRING"}]},
    {"label": "Habilidade", "properties": []},
    {"label": "Conhecimento_Tecnico", "properties": []},
    {"label": "Recurso", "properties": [{"name": "nome", "type": "STRING"}, {"name": "categoria", "type": "STRING"}]},
    {"label": "Tecnologia", "properties": []},
    {"label": "Framework", "properties": []},
    {"label": "Qualificacao", "properties": [{"name": "nome", "type": "STRING"}, {"name": "orgao_emissor", "type": "STRING"}]},
    {"label": "Certificacao", "properties": []},
    {"label": "Treinamento", "properties": []},
    {"label": "Projeto", "properties": [{"name": "nome", "type": "STRING"}, {"name": "area_negocio", "type": "STRING"}, {"name": "status", "type": "STRING"}]},
    {"label": "Idioma", "properties": [{"name": "nome", "type": "STRING"}]},
]

# ------------------ Relationship Types ------------------
RELATIONSHIP_TYPES = [
    {"label": "IS_A", "description": "Relação hierárquica"},
    {"label": "TEM_COMPETENCIA", "properties": [{"name": "nivel_proficiencia", "type": "STRING"}, {"name": "anos_experiencia", "type": "INTEGER"}]},
    {"label": "USA_RECURSO", "properties": [{"name": "nivel_proficiencia", "type": "STRING"}, {"name": "ultima_utilizacao", "type": "INTEGER"}]},
    {"label": "PARTICIPOU_DE", "properties": [{"name": "data_inicio", "type": "DATE"}, {"name": "data_fim", "type": "DATE"}, {"name": "papel_exercido", "type": "STRING"}]},
    {"label": "TEM_QUALIFICACAO", "properties": [{"name": "data_obtencao", "type": "DATE"}, {"name": "expiracao", "type": "DATE"}]},
    {"label": "FALA_IDIOMA", "properties": [{"name": "fluencia", "type": "STRING"}]},
    {"label": "TRABALHOU_EM", "properties": [{"name": "data_entrada", "type": "DATE"}, {"name": "data_saida", "type": "DATE"}]},
    {"label": "REQUER_COMPETENCIA"},
    {"label": "REQUER_RECURSO"},
    {"label": "E_RELACIONADA_A", "description": "Relação horizontal entre competências"},
]

# ------------------ Patterns ------------------
PATTERNS = [
    ("Profissional", "IS_A", "Pessoa"),
    ("Empresa", "IS_A", "Organização"),
    ("Cliente", "IS_A", "Organização"),
    ("Habilidade", "IS_A", "Competencia"),
    ("Conhecimento_Tecnico", "IS_A", "Competencia"),
    ("Tecnologia", "IS_A", "Recurso"),
    ("Framework", "IS_A", "Recurso"),
    ("Certificacao", "IS_A", "Qualificacao"),
    ("Treinamento", "IS_A", "Qualificacao"),
    ("Pessoa", "TEM_COMPETENCIA", "Competencia"),
    ("Pessoa", "USA_RECURSO", "Recurso"),
    ("Pessoa", "PARTICIPOU_DE", "Projeto"),
    ("Pessoa", "TEM_QUALIFICACAO", "Qualificacao"),
    ("Pessoa", "FALA_IDIOMA", "Idioma"),
    ("Pessoa", "TRABALHOU_EM", "Organização"),
    ("Projeto", "REQUER_COMPETENCIA", "Competencia"),
    ("Projeto", "REQUER_RECURSO", "Recurso"),
    ("Competencia", "E_RELACIONADA_A", "Competencia"),
]

prompt_template = """
    You are a top-tier algorithm designed for extracting structured information
    from professional profiles and résumés (CVs) to build a specialized
    **Human Resources (HR) Knowledge Graph**.

    Your **primary objective** is to extract the entities (nodes) and specify their type
    from the following text, and then establish relationships between them.
    Your **highest priority** is to always establish a relationship between the
    main 'Person' entity in the text and all other extracted entities (e.g., Skill,
    Project, Organization, Qualification).

    Extract the entities (nodes) and specify their type from the following text.
    Also extract the relationships between these nodes.

    Return result as JSON using the following format:
    {{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
    "relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

    Use only the following node and relationship types (if provided):
    {schema}

    Assign a unique ID (string) to each node, and reuse it to define relationships.
    Do respect the source and target node types for relationship and
    the relationship direction.

    Make sure you adhere to the following rules to produce valid JSON objects:
    - Do not return any additional information other than the JSON in it.
    - Omit any backticks around the JSON - simply output the JSON on its own.
    - The JSON object must not wrapped into a list - it is its own JSON object.
    - Property names must be enclosed in double quotes

    Examples:
    {examples}

    Input text:

    {text}
"""

# ------------------ SimpleKGPipeline ------------------
pipeline = SimpleKGPipeline(
    driver=driver,
    llm=llm,
    embedder=embedder,
    schema={
        "node_types": NODE_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
        "patterns": PATTERNS,
        "additional_node_types": False,
    },
    prompt_template=prompt_template,
)

# ------------------ PDF Processing ------------------
async def run_pipeline_on_file(file_path, pipeline):
    await pipeline.run_async(file_path=file_path)

nest_asyncio.apply()

ROOT = Path(__file__).resolve().parents[1]  # raiz do repo
PDF_DIR = ROOT / "pdfs"
PDF_DIR.mkdir(exist_ok=True)
pdf_files = [str(p) for p in PDF_DIR.glob("*.pdf")]

if not pdf_files:
    print(f"❌ Nenhum arquivo PDF encontrado em {PDF_DIR}.")
else:
    for file_path in pdf_files:
        print(f"✅ Executando pipeline em: {file_path}")
        asyncio.run(run_pipeline_on_file(file_path, pipeline))

# ------------------ Index Vetorial ------------------
create_vector_index(driver, name="chunkEmbeddings", label="Chunk", embedding_property="embedding", dimensions=1536, similarity_fn="cosine")
