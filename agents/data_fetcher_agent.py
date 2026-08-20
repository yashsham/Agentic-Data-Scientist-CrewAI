# agents/data_fetcher_agent.py

from crewai import Agent, Task
from crewai.tools import tool
import pandas as pd
import os
import shutil


@tool("CSV Ingestion Tool")
def csv_ingestion_tool(file_path: str) -> str:
    """
    Reads any CSV file from disk, creates a working copy at 'data/cleaned_dataset.csv',
    and outputs structural statistics, column info, missing values, and data preview.
    - file_path: Path to the input CSV file.
    """
    clean_path = str(file_path).strip().strip("'").strip('"')
    if not os.path.exists(clean_path):
        return f"Error: File '{clean_path}' does not exist."
    try:
        df = pd.read_csv(clean_path)
        
        # Save working copy for pipeline
        os.makedirs("data", exist_ok=True)
        working_path = os.path.join("data", "cleaned_dataset.csv")
        df.to_csv(working_path, index=False)
        
        summary = f"Successfully loaded dataset: {clean_path}\n"
        summary += f"Working file saved at: {working_path}\n"
        summary += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        summary += f"Column Data Types: {dict(df.dtypes.astype(str))}\n"
        summary += f"Missing Values per Column: {dict(df.isnull().sum())}\n"
        summary += f"\nDataset Summary Statistics:\n{df.describe(include='all').to_string()[:1000]}\n"
        summary += f"\nData Head (First 5 Rows):\n{df.head(5).to_string()}\n"
        return summary
    except Exception as e:
        return f"Error ingesting CSV file '{clean_path}': {e}"


class DataFetcherAgents:
    def make_data_fetcher_agent(self, llm):
        return Agent(
            role='Expert Data Fetcher',
            goal='Ingest dataset from disk using CSV Ingestion Tool, create working dataset file, and pass structural summary downstream.',
            backstory='A data engineer skilled in loading large CSV files efficiently and summarizing dataset metrics.',
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[csv_ingestion_tool]
        )

    def make_fetch_task(self, agent, file_path):
        return Task(
            description=(
                f'Use the CSV Ingestion Tool to load the dataset from path "{file_path}".\n'
                'Output the complete summary including file path, dimensions, column names, data types, missing value statistics, and preview.'
            ),
            expected_output='A complete structural summary of the ingested CSV dataset.',
            agent=agent
        )