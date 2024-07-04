import os
import pandas as pd
import sqlite3
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from typing import TypedDict, Optional
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys and configuration
LLAMA_API = os.getenv("LLAMA_API_KEY")
LANGSMITH_API = os.getenv("LANGSMITH_API_KEY")

if not LLAMA_API:
    raise ValueError("LLAMA_API_KEY is not set in the environment variables.")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API or ""
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "RetailX_AI_Assistant")

# ... [rest of the data preparation and DB_DESCRIPTION remains the same] ...

# Model initialization
model = ChatOpenAI(
    openai_api_key=LLAMA_API,
    openai_api_base=os.getenv("LLAMA_API_BASE", "https://api.llama-api.com"),
    model=os.getenv("LLAMA_MODEL", "llama3-70b")
)

# Pydantic model for CanAnswer
class CanAnswer(BaseModel):
    reasoning: str = Field(description="Reasoning for whether the question can be answered")
    can_answer: bool = Field(description="Whether the question can be answered")

# Workflow steps
can_answer_parser = PydanticOutputParser(pydantic_object=CanAnswer)

can_answer_router_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \n

    {data_description} \n\n

    Given the user's question, decide whether the question can be answered using the information in the database. \n\n

    {format_instructions}

    user
    Question: {question} \n
    assistant""",
    input_variables=["data_description", "question"],
    partial_variables={"format_instructions": can_answer_parser.get_format_instructions()}
)

can_answer_router = can_answer_router_prompt | model | can_answer_parser

def check_if_can_answer_question(state):
    logger.info("Checking if question can be answered")
    result = can_answer_router.invoke({"question": state["question"], "data_description": DB_DESCRIPTION})
    logger.info(f"Can answer result: {result}")
    return {"plan": result.reasoning, "can_answer": result.can_answer}

def skip_question(state):
    return "no" if state.get("can_answer", False) else "yes"

# ... [write_query_prompt remains the same] ...

write_query_chain = write_query_prompt | model | (lambda x: x.strip() if x else "")

def write_query(state):
    logger.info("Writing SQL query")
    result = write_query_chain.invoke({
        "data_description": DB_DESCRIPTION,
        "question": state["question"],
        "plan": state.get("plan", "No plan available")
    })
    logger.info(f"SQL query: {result}")
    return {"sql_query": result}

def execute_query(state):
    logger.info("Executing SQL query")
    query = state.get("sql_query")
    if not query:
        logger.error("No SQL query available to execute")
        return {"sql_result": "Error: No SQL query to execute"}
    try:
        result = query_db(query).to_markdown()
        logger.info(f"Query result: {result}")
        return {"sql_result": result}
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return {"sql_result": f"Error: {str(e)}"}

# ... [write_answer_prompt remains the same] ...

write_answer_chain = write_answer_prompt | model | (lambda x: x.strip() if x else "")

def write_answer(state):
    logger.info("Writing answer")
    result = write_answer_chain.invoke({
        "question": state["question"],
        "plan": state.get("plan", "No plan available"),
        "sql_result": state.get("sql_result", "No results available"),
        "sql_query": state.get("sql_query", "No query available")
    })
    logger.info(f"Answer: {result}")
    return {"answer": result}

# ... [cannot_answer_prompt remains the same] ...

cannot_answer_chain = cannot_answer_prompt | model | (lambda x: x.strip() if x else "")

def explain_no_answer(state):
    logger.info("Explaining why question cannot be answered")
    result = cannot_answer_chain.invoke({
        "problem": state.get("plan", "Unknown problem"),
        "question": state["question"]
    })
    logger.info(f"Explanation: {result}")
    return {"answer": result}

# Workflow state definition
class WorkflowState(TypedDict):
    question: str
    plan: Optional[str]
    can_answer: bool
    sql_query: Optional[str]
    sql_result: Optional[str]
    answer: Optional[str]

# ... [rest of the workflow setup remains the same] ...

# Test the workflow
if __name__ == "__main__":
    inputs = {"question": "Count customers by zip code. Return the 5 most common zip codes"}
    logger.info(f"Invoking app with inputs: {inputs}")
    try:
        result = app.invoke(inputs)
        logger.info(f"App result: {result}")
        print(result)
    except Exception as e:
        logger.error(f"Error running the workflow: {str(e)}")
        print(f"An error occurred: {str(e)}")