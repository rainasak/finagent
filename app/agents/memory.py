from typing import List, Dict, Any
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from app.utils.logging import setup_logger

class MemoryManager:
    """Manages conversation memory for the agent system."""
    
    def __init__(self, session_id: str):
        self.logger = setup_logger(f"{__name__}.MemoryManager")
        self.session_id = session_id
        
        # Initialize in-memory conversation buffer
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Initialize message summarizer
        self.summarizer = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            temperature=0
        )
        
        # Initialize vector store for semantic search
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.message_count = 0  # Track number of messages for indexing
        
        self.recent_context = []  # Store recent interactions
        self.max_recent_interactions = 5  # Keep last 5 interactions
        self.summary = ""  # Store running summary of conversation
        
        self.logger.info("Initialized MemoryManager with semantic search capability")
        
    def _initialize_vector_store(self):
        """Initialize the vector store with existing messages."""
        try:
            messages = self.get_chat_history()
            if not messages:
                # Create empty vector store if no messages
                self.vector_store = FAISS.from_texts(
                    ["initial_empty_text"], 
                    self.embeddings,
                )
                return
            
            # Create text chunks from messages
            texts = [f"{msg['role']}: {msg['content']}" for msg in messages]
            metadatas = [{"role": msg["role"]} for msg in messages]
            
            # Initialize vector store
            self.vector_store = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=metadatas
            )
            self.message_count = len(messages)
            # self.logger.debug(f"Initialized vector store with {self.message_count} messages")
            
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            
    def _update_vector_store(self, role: str, content: str):
        """Update vector store with a new message."""
        try:
            if self.vector_store is None:
                self._initialize_vector_store()
                return
            
            # Add only the new message
            text = f"{role}: {content}"
            metadata = {"role": role}
            
            self.vector_store.add_texts(
                [text],
                metadatas=[metadata]
            )
            self.message_count += 1
            # self.logger.debug("Added new message to vector store")
            
        except Exception as e:
            self.logger.error(f"Error updating vector store: {str(e)}")
            
    def _summarize_context(self) -> str:
        """Create or update the running summary of the conversation."""
        try:
            if not self.recent_context:
                return ""
                
            # If we already have a summary, include it for context
            prompt = "Summarize the following conversation"
            if self.summary:
                prompt += ", incorporating this previous summary:\n" + self.summary + "\n\nNew messages:"
            
            # Add recent messages to summarize
            conversation = "\n".join(
                f"{msg['role']}: {msg['content']}" 
                for msg in self.recent_context
            )
            
            response = self.summarizer.invoke(
                prompt + "\n" + conversation + "\n\nProvide a concise summary that preserves key information and context."
            )
            
            self.summary = response.content
            return self.summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing context: {str(e)}")
            return self.summary  # Return existing summary on error

    def add_to_memory(self, role: str, content: str) -> None:
        """Add an interaction to memory."""
        # Create message
        message = AIMessage(content=content) if role == "assistant" else HumanMessage(content=content)
        
        # Add to conversation memory
        self.conversation_memory.chat_memory.add_message(message)
        
        # Update recent context
        self.recent_context.append({"role": role, "content": content})
        if len(self.recent_context) > self.max_recent_interactions:
            # Summarize and update context before removing old messages
            self._summarize_context()
            self.recent_context.pop(0)  # Remove oldest interaction
        
        # Update vector store for semantic search
        self._update_vector_store(role, content)
        
        # self.logger.debug(f"Added {role} message to memory and updated indexes")

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Retrieve the complete chat history."""
        messages = []
        for msg in self.conversation_memory.chat_memory.messages:
            role = "assistant" if isinstance(msg, AIMessage) else "human"
            messages.append({"role": role, "content": msg.content})
        return messages

    def get_relevant_context(self, query: str) -> List[Dict[str, str]]:
        """Retrieve relevant context using semantic search."""
        try:
            relevant_messages = []
            
            # Always include running summary if available
            if self.summary:
                relevant_messages.append({
                    "role": "system",
                    "content": f"Previous conversation summary: {self.summary}"
                })
            
            # Include recent context
            relevant_messages.extend(self.recent_context)
            
            # Perform semantic search if vector store exists
            if self.vector_store is not None:
                search_results = self.vector_store.similarity_search(
                    query,
                    k=3  # Get top 3 most relevant messages
                )
                
                # Add semantically relevant messages
                for doc in search_results:
                    message = {
                        "role": doc.metadata["role"],
                        "content": doc.page_content.split(": ", 1)[1]  # Remove role prefix
                    }
                    if message not in relevant_messages:
                        relevant_messages.append(message)
            
            # self.logger.debug(f"Found {len(relevant_messages)} relevant messages for context")
            return relevant_messages
            
        except Exception as e:
            self.logger.error(f"Error getting relevant context: {str(e)}")
            return self.recent_context  # Fall back to recent context only        