import os
import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

# Toolkit imports
from langchain_google_community import GmailToolkit, CalendarToolkit

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Use the stable import that matches your current environment
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# --- THE WRAPPER FUNCTION ---
# This allows backend.py to import the agent logic safely without 
# triggering the Google login flow immediately.
def get_agent_app():
    print("Initializing PlanX Agent components...")
    
    # 1. Get the real date dynamically
    current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")

    system_prompt = f"""
    You are planX, a smart assistant for a Developer/Student.
    Current Date: {current_date}
    Your Goal: Automate daily tasks to save time.

    Capabilities:
    1. GMAIL: Check unread emails, send replies, delete promotions.
    2. CALENDAR: Check schedule, book meetings, find free slots.
    3. SEARCH: Search real-time information.

    ⚠️ CRITICAL RULES FOR TOOL USE:
    1. You MUST use valid JSON format for all tool arguments.
    2. You MUST use DOUBLE QUOTES for all JSON keys and values. 
       - CORRECT: {{"calendar_id": "primary", "time": "10:00"}}
       - WRONG: {{'calendar_id': 'primary', 'time': '10:00'}}
    3. If checking "today's events", calculate the date from Current Date: {current_date}.
    """

    # 2. Load Tools Safely
    # If token.json is missing, we catch the error so the server can still start
    try:
        print("   - Loading Google Toolkits...")
        gmail_tools = GmailToolkit().get_tools()
        calendar_tools = CalendarToolkit().get_tools()
    except Exception as e:
        print(f"⚠️ Warning: Google Tools failed to load (Auth needed): {e}")
        gmail_tools = []
        calendar_tools = []

    search_tool = TavilySearchResults(max_results=3)
    master_tools = gmail_tools + calendar_tools + [search_tool]

    # 3. Setup LLM (Corrected Model Name)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Fixed from 2.5
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0
    )

    # 4. Bind tools and build the graph
    llm_with_tools = llm.bind_tools(master_tools)

    def agent_node(state: MessagesState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(master_tools)

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    # 5. Return both the compiled app and the system prompt
    return workflow.compile(), system_prompt

# --- CLI MODE ---
if __name__ == "__main__":
    print("Starting PlanX CLI Mode...")
    try:
        app, sys_prompt = get_agent_app()
        current_state = {"messages": [SystemMessage(content=sys_prompt)]}

        print("\n" + "="*40 + "\nPlanX v1.0 is online\n" + "="*40 + "\n")

        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            current_state["messages"].append(HumanMessage(content=user_input))
            final_state = app.invoke(current_state)
            
            ai_response = final_state["messages"][-1]
            content = ai_response.content
            
            final_text = ""
            if isinstance(content, str):
                final_text = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        final_text += part["text"]
                    elif isinstance(part, str):
                        final_text += part
            
            print(f"\nPlanX: {final_text or 'Task completed.'}\n")
            current_state = final_state
    except Exception as e:
        print(f"Critical Error: {e}")
