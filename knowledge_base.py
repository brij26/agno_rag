from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.lancedb import LanceDb
from agno.knowledge.knowledge import Knowledge
from dotenv import load_dotenv

# load the api keys
load_dotenv()

# define the embedder
embedder = OpenAIEmbedder()

# Define the chunking stratergy
chunking_strategy = SemanticChunking(embedder=embedder, chunk_size=1000)

# create the pdf reader
reader = PDFReader(chunking_strategy=chunking_strategy)

# Create the vector db
vector_db = LanceDb(uri="vectordb/lancedb",
                    table_name="Knowledge_table",
                    embedder=embedder
                    )

# define the knowledge base
knowledge_base = Knowledge(name="Knowledge_Base",
                           description="It contains research paper 'attention is all you need'",
                           vector_db=vector_db)


if __name__ == "__main__":
    # add content to the knowledge base
    knowledge_base.add_content(path="attention_is_all_you_need.pdf",
                               reader=reader)
