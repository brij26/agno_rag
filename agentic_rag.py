from agno.agent import Agent
from agno.models.openai import OpenAIChat
from knowledge_base import knowledge_base
from dotenv import load_dotenv

# load all the keys to the env
load_dotenv()

# define model
llm = OpenAIChat(id="gpt-4o-mini")

# define agent
agent = Agent(
    model=llm,
    name="Knowledge agent",
    knowledge=knowledge_base,
    search_knowledge=True,
    instructions=["You are helpful assistnace",
                  "If anyone ask you about transformer model you can refer to knowledge base to answer the query",
                  "Try not to hallucinate while answering the quesiton",
                  "If you don't know the answer of any query then say i don't know"],
    stream=True,
    markdown=True
)


agent.print_response(input="Can you tell me what is the capital of india?")

agent.print_response(
    input="Tell me about self attention layer of the transformer in 100 words")
