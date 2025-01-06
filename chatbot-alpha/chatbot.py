from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import streamlit as st

api_key='<API_KEY>'

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            return value["messages"][-1].content


# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def update_chat_history(user_input, assistant_response):
  """Updates the chat history with user input and assistant response"""
  st.session_state.chat_history.append({"role": "user", "content": user_input})
  st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

def display_chat_history():
  """Displays the chat history in the Streamlit app"""
  for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

st.title("Lang graph - gemini - chatbot")

# Display chat history on app launch
display_chat_history()

# Get user input
user_input = st.chat_input(placeholder="Type your message here...")

# Update chat history with user input
if user_input:
  assistant_response = stream_graph_updates(user_input)
  update_chat_history(user_input, assistant_response)

# Display updated chat history
display_chat_history()

## run using the command :: streamlit run chatbot.py