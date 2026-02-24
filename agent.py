"""
agent.py
────────
Intelligence agent for AI-BOS.
Uses LangChain's create_tool_calling_agent with Gemini-2.0-flash.
Tools:
  - Retrieval: Accesses the knowledge base.
  - PythonREPLTool: Executes Python code.
  - Calculator: Performs complex math.
  - SQLStub: Interacts with a local SQLite database.
"""
import os
import sqlite3
import json
from typing import Any, List, Optional, Union, Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, StructuredTool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain.memory import ConversationSummaryBufferMemory
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.tools import PythonREPLTool

from rag import build_advanced_retriever, ingest_documents_async, load_and_chunk_files
from utils import get_logger

logger = get_logger(__name__)

# --- Persistence Helpers ---

class JSONFileChatMessageHistory(BaseChatMessageHistory):
    """Simple JSON-based chat history persistence."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.messages: List[BaseMessage] = []
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    self.messages = messages_from_dict(data)
            except Exception as e:
                logger.error(f"Error loading chat history: {e}")

    def save(self):
        try:
            with open(self.file_path, "w") as f:
                json.dump([message_to_dict(m) for m in self.messages], f)
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self.save()

    def clear(self) -> None:
        self.messages = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

# --- Structured Models ---

class KPI(BaseModel):
    """Key Performance Indicator model."""
    name: str = Field(description="Name of the KPI")
    value: Union[float, int] = Field(description="Current value")
    target: Union[float, int] = Field(description="Target value")
    unit: str = Field(description="Metric unit (e.g., USD, %, count)")
    status: str = Field(description="Status (e.g., On Track, At Risk)")

class SWOT(BaseModel):
    """SWOT Analysis model."""
    strengths: List[str] = Field(description="Internal positive factors")
    weaknesses: List[str] = Field(description="Internal negative factors")
    opportunities: List[str] = Field(description="External positive factors")
    threats: List[str] = Field(description="External negative factors")

def get_structured_output(llm: Any, schema: type[BaseModel], prompt: str) -> BaseModel:
    """
    Invokes the LLM with a specific output schema and retry logic for validation errors.
    """
    structured_llm = llm.with_structured_output(schema)
    try:
        return structured_llm.invoke(prompt)
    except Exception as e:
        logger.warning(f"Validation error on first attempt: {e}. Retrying once...")
        try:
            # Simple retry logic
            return structured_llm.invoke(f"There was a validation error: {e}. Please ensure the output strictly follows the schema. {prompt}")
        except Exception as retry_e:
            logger.error(f"Structured output failed after retry: {retry_e}")
            raise retry_e

@tool
def generate_business_summary(analysis_type: str, context: str) -> str:
    """
    Useful for generating structured SWOT or KPI summaries.
    analysis_type should be 'SWOT' or 'KPI'.
    context should be the raw data to analyze.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    # Low-latency model for extraction
    extractor_llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=api_key)
    
    try:
        if analysis_type.upper() == "SWOT":
            result = get_structured_output(extractor_llm, SWOT, f"Analyze the following context and provide a SWOT analysis: {context}")
            return json.dumps(result.dict() if hasattr(result, 'dict') else result.model_dump(), indent=2)
        elif analysis_type.upper() == "KPI":
            result = get_structured_output(extractor_llm, KPI, f"Extract the primary KPI from the following context: {context}")
            return json.dumps(result.dict() if hasattr(result, 'dict') else result.model_dump(), indent=2)
        else:
            return "Invalid analysis type. Use SWOT or KPI."
    except Exception as e:
        return f"Structured Analysis Error: {e}"

# --- Tool Definitions ---

@tool
def calculator(expression: str) -> str:
    """Useful for when you need to answer questions about math."""
    try:
        # Simple eval for math expressions (safe for demo, but limited)
        # In production, use a more robust math parser
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def sql_query_tool(query: str) -> str:
    """Useful for interacting with the local SQLite database 'intel_forge.db'."""
    try:
        conn = sqlite3.connect("intel_forge.db")
        cursor = conn.cursor()
        cursor.execute(query)
        if query.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            conn.close()
            return str(results)
        else:
            conn.commit()
            conn.close()
            return "Query executed successfully."
    except Exception as e:
        return f"Database error: {e}"

def get_retrieval_tool(llm: Any, docs: List[Any]):
    """Creates a retrieval tool from the RAG pipeline."""
    retriever = build_advanced_retriever(llm, "ai-bos", docs)
    
    @tool
    def retrieval_tool(query: str) -> str:
        """Useful for answering questions about business documents, plans, and IntelForge architecture."""
        try:
            # We use invoke here for synchronous tool execution
            # The retriever is a ContextualCompressionRetriever
            results = retriever.get_relevant_documents(query)
            return "\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            return f"Retrieval error: {e}"
            
    return retrieval_tool

# --- Agent Factory ---

def build_business_agent(docs: List[Any], chat_history_file: str = "chat_history.json"):
    """Builds and returns the AgentExecutor with persistent memory."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found in environment.")
        
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0
    )
    
    # 1. Initialize Tools
    tools = [
        get_retrieval_tool(llm, docs),
        PythonREPLTool(),
        calculator,
        sql_query_tool,
        generate_business_summary
    ]
    
    # 2. Define Memory (Hybrid: Buffer + Summary)
    # Persist via JSONFileChatMessageHistory
    history = JSONFileChatMessageHistory(chat_history_file)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=2000,
        memory_key="chat_history",
        return_messages=True,
        chat_memory=history
    )
    
    # 3. Define Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the IntelForge Intelligent Business Agent. "
                   "You have access to tools for retrieval, Python execution, math, and SQL. "
                   "Use the chat_history to maintain context in multi-turn conversations."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 4. Create Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 5. Agent Executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory,
        verbose=True, 
        handle_parsing_errors=True
    )
    
    return agent_executor
