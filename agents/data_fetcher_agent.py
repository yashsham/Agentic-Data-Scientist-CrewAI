# agents/data_fetcher_agent.py

from crewai import Agent, Task
import pandas as pd


class DataFetcherAgents:
    def make_data_fetcher_agent(self, llm):
        return Agent(
            role='Expert Data Fetcher',
            goal='Fetch a dataset from a specified file path, load it, and output a structured overview with a JSON representation.',
            backstory='An expert in data ingestion and dataset analysis who provides clear data structures for downstream tasks.',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    def make_fetch_task(self, agent, file_path):
        return Task(
            description=(
                f'Load the dataset from file path "{file_path}".\n'
                'Inspect the dataset shape, column names, data types, and first few rows.\n'
                'Output a comprehensive text summary including:\n'
                '1. File path and dataset dimensions (rows x columns).\n'
                '2. List of column names and their data types.\n'
                '3. The complete data in JSON format so subsequent agents can process it.'
            ),
            expected_output='A dataset summary including dimensions, column details, and the full DataFrame in JSON format.',
            agent=agent
        )