import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

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
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content)) 
    print(f"\nAI: {response.content}")
    print("\n<-------------------------------------------->\n")
    print("CURRENT STATE: ", state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")


with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")