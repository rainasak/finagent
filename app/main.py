import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from app.agents.core import FinancialAgent
from app.utils.logging import setup_logger

# Set up logger for this module
logger = setup_logger(__name__)

# Load environment variables

# Configure page
st.set_page_config(
    page_title="FinAgent Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger.debug("Streamlit page configured")

# Initialize session state for messages if not exists
if 'messages' not in st.session_state:
    st.session_state.messages = []
    logger.debug("Initialized empty messages in session state")
if 'financial_agent' not in st.session_state:
    logger.info("Initializing FinancialAgent")
    st.session_state['financial_agent'] = FinancialAgent()

# Set up the app title and description
st.title("📊 FinAgent")

# Create sidebar for settings and tools
with st.sidebar:
    st.header("🛠️ Tools and Settings")
    st.markdown("""
    Available capabilities:
    - Web Search: Find the latest financial news and data.
    - Calculator: Perform financial calculations.
    - Code Executor: Run Python code.
    - Document Summarizer: Summarize financial reports and documents.
    """)

# Create a container for chat messages
chat_container = st.container()

# Create a container for the input form at the bottom
with st.container():
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6,1])
        with col1:
            user_input = st.text_input(
                "Enter your financial analysis query:",
                placeholder="e.g., Analyze Apple's Q3 2023 earnings and predict short-term trends",
                key="user_input",
                label_visibility="collapsed"
            )
        with col2:
            submit = st.form_submit_button("🔍 Analyze", use_container_width=True)

# Handle form submission
if submit and user_input.strip():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get AI response
    with st.spinner("Analyzing..."):
        response = st.session_state['financial_agent'].process_query(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat messages in the chat container
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style='padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem;'>
                        <b>You:</b> {message["content"]}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:            # Handle assistant messages with potential plots/visualizations
            content = message["content"]
            if isinstance(content, dict):
                if "display" in content and content['display']:
                    with st.container(border=True):
                # For responses with display field (plots, formatted text)
                        st.markdown(
                            f"""
                            <div style='padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem;'>
                                <b>🤖 FinAgent:</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.markdown(content["display"], unsafe_allow_html=True)
                if "content" in content and content['content']:
                    # For regular text responses
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style='padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem;'>
                                <b>🤖 FinAgent:</b> {content['content'].replace('$', r'\$')}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )