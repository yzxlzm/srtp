from langchain.agents import AgentType, initialize_agent

from agent_tools import TOOLS
from llm import get_llm


def build_agent():
    llm = get_llm()
    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
    return agent

