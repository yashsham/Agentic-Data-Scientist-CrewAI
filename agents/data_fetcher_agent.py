# agents/data_fetcher_agent.py

from crewai import Agent, Task
from crewai.tools import tool
import pandas as pd
import os


@tool("CSV Reader Tool")
def csv_reader_tool(file_path: str) -> str:
    """
    Reads a CSV file from the local filesystem and returns dataset shape, column information, summary stats, and JSON data.
    - file_path: Path to the CSV file to read.
    """
    clean_path = str(file_path).strip().strip("'").strip('"')
    if not os.path.exists(clean_path):
        return f"Error: File '{clean_path}' does not exist."
    try:
        df = pd.read_csv(clean_path)
        summary = f"Dataset File: {clean_path}\n"
        summary += f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns\n"
        summary += f"Columns & Types: {dict(df.dtypes.astype(str))}\n"
        summary += f"Missing Value Counts: {dict(df.isnull().sum())}\n"
        summary += f"Dataset Preview:\n{df.head(10).to_string()}\n\n"
        summary += f"DataFrame JSON:\n{df.to_json()}"
        return summary
    except Exception as e:
        return f"Error reading CSV file '{clean_path}': {e}"


class DataFetcherAgents:
    def make_data_fetcher_agent(self, llm):
        return Agent(
            role='Expert Data Fetcher',
            goal='Fetch a dataset from a specified file path using the CSV Reader Tool, inspect its structure, and pass the data to downstream agents.',
            backstory='An expert in data ingestion who uses specialized file tools to load datasets into memory and extract structural metadata.',
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[csv_reader_tool]
        )

    def make_fetch_task(self, agent, file_path):
        return Task(
            description=(
                f'Use the CSV Reader Tool to load the CSV dataset from path "{file_path}".\n'
                'Extract the dataset shape, column names, data types, missing value statistics, and the full DataFrame in JSON format.\n'
                'Output the complete tool output so downstream cleaning, visualization, and report agents have access to the dataset.'
            ),
            expected_output='The dataset summary, column statistics, preview, and complete DataFrame JSON representation.',
            agent=agent
        )