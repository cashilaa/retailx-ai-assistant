import os
from dotenv import load_dotenv
import pandas as pd
import sqlite3
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
import logging

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API key configuration
LLAMA_API = os.getenv("LLAMA_API")
LANGSMITH_API = os.getenv("LANGSMITH_API")

# Data preparation
data = {
    "Customer_ID": [1, 2, 3, 4, 5],
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Gender": ["Female", "Male", "Male", "Male", "Female"],
    "Age": [25, 30, 35, 40, 45],
    "Country": ["USA", "USA", "Canada", "Canada", "USA"],
    "State": ["CA", "NY", "BC", "ON", "TX"],
    "City": ["Los Angeles", "New York", "Vancouver", "Toronto", "Houston"],
    "Zip_Code": ["90001", "10001", "V5K0A1", "M5H2N2", "77001"],
    "Product": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"],
    "Category": ["Electronics", "Electronics", "Electronics", "Electronics", "Accessories"],
    "Price": [1200, 800, 600, 300, 100],
    "Purchase_Date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-22"],
    "Quantity": [1, 2, 3, 1, 4],
    "Total_Spent": [1200, 1600, 1800, 300, 400]
}

df = pd.DataFrame(data)

# Save to SQLite database
conn = sqlite3.connect('retail.db')
df.to_sql('Retail', conn, if_exists='replace', index=False)
conn.close()

# Database description
DB_DESCRIPTION = """You have access to the following tables and columns in a SQLite3 database:

Retail Table
Customer_ID: A unique ID that identifies each customer.
Name: The customer's name.
Gender: The customer's gender: Male, Female.
Age: The customer's age.
Country: The country where the customer resides.
State: The state where the customer resides.
City: The city where the customer resides.
Zip_Code: The zip code where the customer resides.
Product: The product purchased by the customer.
Category: The category of the product.
Price: The price of the product.
Purchase_Date: The date when the purchase was made.
Quantity: The quantity of the product purchased.
Total_Spent: The total amount spent by the customer.
"""

# Query execution function
def query_db(query):
    conn = sqlite3.connect('retail.db')
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()

# Model initialization
model = ChatOpenAI(
    openai_api_key=LLAMA_API,
    openai_api_base="https://api.llama-api.com",
    model="llama3-70b"
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
    if state["can_answer"]:
        return "no"
    else:
        return "yes"

write_query_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \n

    {data_description} \n\n

    In the previous step, you have prepared the following plan: {plan}

    Return an SQL query with no preamble or explanation. Don't include any markdown characters or quotation marks around the query.
    user
    Question: {question} \n
    assistant""",
    input_variables=["data_description", "question", "plan"],
)

class SimpleStrOutputParser:
    def parse(self, text):
        return text.strip()

write_query_chain = write_query_prompt | model | SimpleStrOutputParser()

def write_query(state):
    logger.info("Writing SQL query")
    result = write_query_chain.invoke({
        "data_description": DB_DESCRIPTION,
        "question": state["question"],
        "plan": state["plan"]
    })
    logger.info(f"SQL query: {result}")
    return {"sql_query": result}

def execute_query(state):
    logger.info("Executing SQL query")
    query = state["sql_query"]
    try:
        result = query_db(query).to_markdown()
        logger.info(f"Query result: {result}")
        return {"sql_result": result}
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return {"sql_result": str(e)}

write_answer_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \n

    In the previous step, you have planned the query as follows: {plan},
    generated the query {sql_query}
    and retrieved the following data:
    {sql_result}

    Return a text answering the user's question using the provided data.
    user
    Question: {question} \n
    assistant""",
    input_variables=["question", "plan", "sql_query", "sql_result"],
)

write_answer_chain = write_answer_prompt | model | SimpleStrOutputParser()

def write_answer(state):
    logger.info("Writing answer")
    result = write_answer_chain.invoke({
        "question": state["question"],
        "plan": state["plan"],
        "sql_result": state["sql_result"],
        "sql_query": state["sql_query"]
    })
    logger.info(f"Answer: {result}")
    return {"answer": result}

cannot_answer_prompt = PromptTemplate(
    template="""system
    You are a database reading bot that can answer users' questions using information from a database. \n

    You cannot answer the user's questions because of the following problem: {problem}.

    Explain the issue to the user and apologize for the inconvenience.
    user
    Question: {question} \n
    assistant""",
    input_variables=["question", "problem"],
)

cannot_answer_chain = cannot_answer_prompt | model | SimpleStrOutputParser()

def explain_no_answer(state):
    logger.info("Explaining why question cannot be answered")
    result = cannot_answer_chain.invoke({
        "problem": state["plan"],
        "question": state["question"]
    })
    logger.info(f"Explanation: {result}")
    return {"answer": result}

# Workflow state definition
class WorkflowState(TypedDict):
    question: str
    plan: str
    can_answer: bool
    sql_query: str
    sql_result: str
    answer: str

# Create and configure the workflow
workflow = StateGraph(WorkflowState)

workflow.add_node("check_if_can_answer_question", check_if_can_answer_question)
workflow.add_node("write_query", write_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("write_answer", write_answer)
workflow.add_node("explain_no_answer", explain_no_answer)

workflow.set_entry_point("check_if_can_answer_question")

workflow
