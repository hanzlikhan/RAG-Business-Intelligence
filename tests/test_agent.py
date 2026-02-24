"""
tests/test_agent.py
───────────────────
Unit tests for the Intelligent Business Agent tools and construction.
"""

import pytest
import sqlite3
import os
from unittest.mock import MagicMock, patch
from agent import calculator, sql_query_tool, build_business_agent, JSONFileChatMessageHistory, KPI, SWOT, generate_business_summary
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

def test_calculator_valid():
    """Test the calculator tool with a valid expression."""
    result = calculator.invoke("2 + 2 * 5")
    assert result == "12"

def test_calculator_invalid():
    """Test the calculator tool with an invalid expression."""
    result = calculator.invoke("invalid expression")
    assert "Error" in result

def test_sql_query_tool(tmp_path):
    """Test the SQL tool with a temporary database."""
    # We use a mock or a temp db file for testing
    db_path = tmp_path / "intel_forge.db"
    
    # Patch the sqlite3 connect in the tool
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Test SELECT
        mock_cursor.fetchall.return_value = [("Architect",)]
        result = sql_query_tool.invoke("SELECT architect FROM info")
        assert "[('Architect',)]" in result
        
        # Test INSERT/UPDATE
        result = sql_query_tool.invoke("INSERT INTO info VALUES ('Architect')")
        assert "success" in result.lower()

@patch("agent.AgentExecutor")
@patch("agent.ConversationSummaryBufferMemory")
@patch("agent.ChatGoogleGenerativeAI")
@patch("agent.build_advanced_retriever")
def test_agent_construction(mock_retriever, mock_llm, mock_memory, mock_executor):
    """Test that the agent executor is built with the correct tools."""
    # Setup mocks
    mock_retrieval_instance = MagicMock()
    mock_retriever.return_value = mock_retrieval_instance
    
    os.environ["GOOGLE_API_KEY"] = "fake_key"
    
    docs = [Document(page_content="test doc")]
    build_business_agent(docs)
    
    # Verify AgentExecutor was created with a memory object
    mock_executor.assert_called_once()
    _, kwargs = mock_executor.call_args
    assert "memory" in kwargs

def test_json_history_persistence(tmp_path):
    """Test that JSONFileChatMessageHistory correctly saves and loads messages."""
    file_path = str(tmp_path / "test_history.json")
    history = JSONFileChatMessageHistory(file_path)
    
    # Add messages
    history.add_message(HumanMessage(content="Hello"))
    history.add_message(AIMessage(content="Hi there"))
    
    # Reload history
    new_history = JSONFileChatMessageHistory(file_path)
    assert len(new_history.messages) == 2
    assert new_history.messages[0].content == "Hello"
    assert new_history.messages[1].content == "Hi there"

@patch("agent.AgentExecutor")
@patch("agent.ConversationSummaryBufferMemory")
@patch("agent.ChatGoogleGenerativeAI")
@patch("agent.build_advanced_retriever")
def test_agent_memory_initialization(mock_retriever, mock_llm, mock_memory, mock_executor, tmp_path):
    """Test that the agent is initialized with ConversationSummaryBufferMemory."""
    os.environ["GOOGLE_API_KEY"] = "fake_key"
    chat_file = str(tmp_path / "chat.json")
    
    docs = [Document(page_content="test")]
    build_business_agent(docs, chat_history_file=chat_file)
    
    # Check that memory was initialized with correct params
    mock_memory.assert_called_once()
    args, kwargs = mock_memory.call_args
    assert kwargs["max_token_limit"] == 2000
    assert kwargs["memory_key"] == "chat_history"

def test_kpi_model_validation():
    """Test that KPI model correctly validates numeric fields."""
    kpi = KPI(name="Revenue", value=1000.5, target=1200, unit="USD", status="At Risk")
    assert isinstance(kpi.value, (float, int))
    assert kpi.status == "At Risk"

def test_swot_model_validation():
    """Test that SWOT model correctly validates list fields."""
    swot = SWOT(
        strengths=["Fast", "Secure"],
        weaknesses=["New"],
        opportunities=["Growth"],
        threats=["Competition"]
    )
    assert isinstance(swot.strengths, list)
    assert len(swot.strengths) == 2

@patch("agent.ChatGoogleGenerativeAI")
@patch("agent.get_structured_output")
def test_generate_business_summary_tool(mock_structured, mock_llm):
    """Test that generate_business_summary tool returns valid JSON."""
    # Mock SWOT result
    mock_swot = SWOT(strengths=["S"], weaknesses=["W"], opportunities=["O"], threats=["T"])
    mock_structured.return_value = mock_swot
    
    result = generate_business_summary.invoke({"analysis_type": "SWOT", "context": "test data"})
    assert "strengths" in result
    assert "S" in result
