import streamlit as st # Added: Import Streamlit for the web interface
import os
import datetime
from dotenv import load_dotenv

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Added: AIMessage needed for chat history types

# Toolkit imports
from langchain_google_community import GmailToolkit, CalendarToolkit
from langchain_community.tools.tavily_search import TavilySearchResults

# Langgraph imports
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Auth imports (Added for Sidebar Button)
from google_auth_oauthlib.flow import InstalledAppFlow # Added: To handle login flow from UI
from google.oauth2.credentials import Credentials # Added: To handle credentials

# 1. SETUP PAGE CONFIG
st.set_page_config(page_title="PlanX Dashboard", page_icon="🤖") # Added: Configures browser tab title and icon
st.title("🤖 PlanX: AI Assistant") # Added: Sets the main title of the web page

load_dotenv()

# ☁️ CLOUD DEPLOYMENT FIX
# If running on Streamlit Cloud, recreate the secret files from the settings
if "GOOGLE_CREDENTIALS" in st.secrets:
    with open("credentials.json", "w") as f:
        f.write(st.secrets["GOOGLE_CREDENTIALS"])

if "GOOGLE_TOKEN" in st.secrets:
    with open("token.json", "w") as f:
        f.write(st.secrets["GOOGLE_TOKEN"])

# 🔐 SIDEBAR AUTHENTICATION (New Section)

# Added: Define scopes explicitly for the login button
SCOPES = ["https://mail.google.com/", "https://www.googleapis.com/auth/calendar"]

# Added: Function to handle login when button is clicked
def authenticate_google():
    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
    creds = flow.run_local_server(port=0)
    with open("token.json", "w") as token:
        token.write(creds.to_json())
    return True

# Added: Sidebar layout for connection status
with st.sidebar:
    st.header("🔌 Connections") # Added: Sidebar header
    
    # Added: Check if token exists to determine status
    if os.path.exists("token.json"):
        st.success("✅ Google Connected") # Added: Visual success indicator
        if st.button("🔄 Log out"): # Added: Logout button
            os.remove("token.json") # Added: Deletes token on logout
            st.rerun() # Added: Refreshes app to update state
    else:
        st.error("❌ Not Connected") # Added: Visual error indicator
        if st.button("🚀 Connect Google Account"): # Added: Login button
            try:
                authenticate_google() # Added: Trigger auth flow
                st.rerun() # Added: Refresh to show connected state
            except Exception as e:
                st.error(f"Error: {e}") # Added: Error handling

# 🧠 BACKEND SETUP (Cached)

# Added: Cache the agent setup so it doesn't reload on every message (Improves speed)
@st.cache_resource 
def setup_agent():
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
    """

    # Setup llm model
    # --- FIX: Get API Key securely from Streamlit Secrets ---
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = os.environ.get("GEMINI_API_KEY")

    # Setup llm model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=api_key,
        temperature=0
    )

    # Initialize tools (Only if token exists, handled by sidebar check)
    gmail_tools = GmailToolkit().get_tools()
    calendar_tools = CalendarToolkit().get_tools()
    search_tool = TavilySearchResults(max_results=3)

    master_tools = gmail_tools + calendar_tools + [search_tool]

    # Bind tools to the model
    llm_with_tools = llm.bind_tools(master_tools)

    # Define the graph node
    def agent_node(state: MessagesState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Prebuilt tool node
    tool_node = ToolNode(master_tools)

    # Build the graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile(), system_prompt # Added: Return the compiled app and prompt

# Added: Only initialize the agent if we are logged in
if os.path.exists("token.json"):
    app, SYSTEM_PROMPT = setup_agent() # Added: Load the cached agent
    
    # 🎨 CHAT INTERFACE (Replaces while Loop)

    # Added: Initialize session state to store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Added: Loop through history to display previous messages
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"): # Added: User bubble
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            if msg.content: # Added: Only show if there is text content
                with st.chat_message("assistant"): # Added: AI bubble
                    # Added: Handle Gemini's complex list response format
                    content = msg.content
                    final_text = ""
                    if isinstance(content, str):
                        final_text = content
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and "text" in part:
                                final_text += part["text"]
                            elif isinstance(part, str):
                                final_text += part
                    st.write(final_text) # Added: Display cleaned text

    # Added: Input widget at the bottom (Replaces input() function)
    if user_input := st.chat_input("Ask PlanX..."):
        
        # Added: Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Added: Add user message to history
        st.session_state.messages.append(HumanMessage(content=user_input))

        # Added: Display a spinner while AI thinks
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Added: Run the LangGraph agent
                final_state = app.invoke({"messages": st.session_state.messages})
                ai_response = final_state["messages"][-1]

                # Added: Clean up response text (Same logic as above)
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
                
                # Added: Fallback if text is empty
                if not final_text:
                    final_text = "Task completed."

                st.write(final_text) # Added: Display AI response
                
                # Added: Optional - Show tool details in an expander
                if ai_response.tool_calls:
                    with st.expander("🛠️ Tool Usage Details"):
                        st.write(ai_response.tool_calls)

        # Added: Update session state with the new AI message
        st.session_state.messages.append(ai_response)

else:
    st.info("Please connect your Google Account in the sidebar to start.") # Added: Fallback message
