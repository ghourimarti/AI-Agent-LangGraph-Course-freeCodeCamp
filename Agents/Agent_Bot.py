from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values

load_dotenv()

# <----------- Load Environment Variables ----------->

import os
from dotenv import load_dotenv
load_dotenv('token.env')  # path to your token.env file

langchain_api_key = os.getenv("langchain_api_key")
openai_api_keys = os.getenv("openai_api_key")
tavily_api_key = os.getenv("tavily_api_key")
groq_api_key = os.getenv("groq_api_key")

print("Langchain Key:      ",langchain_api_key[:5] + "..." if langchain_api_key else "key not found")
print("openai_api_key Key: ",openai_api_keys[:5] + "..." if openai_api_keys else "key not found")
print("tavily_api_key Key: ",tavily_api_key[:5] + "..." if tavily_api_key else "key not found")
print("groq_api_key Key:   ",groq_api_key[:5] + "..." if groq_api_key else "key not found")


# os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["LANGSMITH_API_KEY"] = langchain_api_key
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Agent Debugging"
# <----------- Load Environment Variables ----------->


class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
