import os
import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

# Toolkit imports
from langchain_google_community import GmailToolkit, CalendarToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Tavily imports
from langchain_community.tools.tavily_search import TavilySearchResults

# # Voice imports
# import speech_recognition as sr
# import pyttsx3

# # Initialize Text-to-Speech Engine
# engine = pyttsx3.init()

# # Optional: Set Voice (0 = Male, 1 = Female usually)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id) # Change index to 0 for Male

# def speak(text):
#     """The AI speaks the text"""
#     print(f"PlanX: {text}") # Print it too
#     engine.say(text)
#     engine.runAndWait()

# def listen():
#     """Listens to the microphone and returns text"""
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening... (Speak now)")
#         try:
#             # Listen for audio (timeout after 5 seconds of silence)
#             audio = r.listen(source, timeout=5, phrase_time_limit=10)
#             print("Thinking...")
#             command = r.recognize_google(audio) # Uses Google's free API
#             print(f"User said: {command}")
#             return command
#         except sr.WaitTimeoutError:
#             return None
#         except sr.UnknownValueError:
#             print("Sorry, I didn't catch that.")
#             return None
#         except sr.RequestError:
#             print("Network error with Speech API.")
#             return None
    
load_dotenv()

# 1. Get the real date dynamically
current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")

system_prompt = f"""
You are planX, a smart assistant for a Developer/Student.
Current Date: {current_date}
Your Goal: Automate daily tasks to save time.

Capabilities:
1. GMAIL: Check unread emails, send replies, delete promotions.
2. CALENDAR: Check schedule, book meetings, find free slots.
3. SEARCH: Search real-time information using search tool.

⚠️ CRITICAL RULES FOR TOOL USE:
1. You MUST use valid JSON format for all tool arguments.
2. You MUST use DOUBLE QUOTES for all JSON keys and values. 
   - CORRECT: {{"calendar_id": "primary", "time": "10:00"}}
   - WRONG: {{'calendar_id': 'primary', 'time': '10:00'}}
3. If checking "today's events", calculate the date from Current Date: {current_date}.
"""
print("PlanX system initializing...")

# Setup llm model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Setup gmail toolkit
print("   - Loading Gmail Toolkit...")
# This looks for 'credentials.json' in your folder automatically
gmail_tools = GmailToolkit().get_tools()

# Get the list of pre-built tools (Read, Send, Search, etc.)
print("   - Loading Calendar Toolkit...")
calendar_tools = CalendarToolkit().get_tools()

# Initialize the search tool
print("   - Loading Tavily Search Tool...")
search_tool = TavilySearchResults(max_results=3)

master_tools = gmail_tools + calendar_tools + [search_tool]

print(f"Tools Ready: {len(master_tools)} loaded.")

# Bind tools to the model
llm_with_tools = llm.bind_tools(master_tools)

# Define the graph node
def agent_node(state: MessagesState):
    """Decides next step based on history."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Prebuilt tool node
tool_node = ToolNode(master_tools)

# Build the graph
workflow = StateGraph(MessagesState)
# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile()

# Initialize Chat Memory with System Prompt
current_state = {
    "messages": [SystemMessage(content=system_prompt)]
}

# # Start with a greeting
# speak("PlanX is online. How can I help you?")

# Only run loop if started directly
if __name__ == "__main__":
    print("Starting PlanX CLI Mode...")
    print("\n" + "="*40)
    print("PlanX v1.0 is online")
    print("(Type 'exit' or 'quit' to stop)")
    print("="*40 + "\n")
    while True:
        try:
            # 1. Get User Input
            user_input = input("User: ")
            
            # 2. Check for Exit
            if user_input.lower() in ["exit", "quit"]:
                print("PlanX: Shutting down. Goodbye!")
                break
            
            # 3. Add User Message to State
            current_state["messages"].append(HumanMessage(content=user_input))

            # 4. Run the Graph
            final_state = app.invoke(current_state)

            # 5. Extract Response (Handling both Strings and Lists)
            ai_response = final_state["messages"][-1]
            content = ai_response.content
            
            final_text = ""

            # Case A: Simple String
            if isinstance(content, str):
                final_text = content
                
            # Case B: Gemini List (Complex response)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        final_text += part["text"]
                    elif isinstance(part, str):
                        final_text += part

            # Fallback if empty
            if not final_text:
                final_text = "Task completed."

            # 6. Print the Response
            print(f"\nPlanX: {final_text}\n")

            # 7. Update Memory
            current_state = final_state

        except Exception as e:
            print(f"Error: {e}")
