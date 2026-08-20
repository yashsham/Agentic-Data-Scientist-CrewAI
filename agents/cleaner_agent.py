# agents/cleaner_agent.py

from crewai import Agent, Task
from crewai.tools import tool
import pandas as pd
import os


@tool("Data Cleaner Tool")
def data_cleaner_tool(csv_path: str = "data/cleaned_dataset.csv") -> str:
    """
    Cleans dataset on disk at 'data/cleaned_dataset.csv'.
    Fills null values in numeric columns with median, fills text nulls with mode/'Unknown',
    and updates the file on disk.
    """
    path = str(csv_path).strip().strip("'").strip('"')
    if not os.path.exists(path):
        path = os.path.join("data", "cleaned_dataset.csv")
    if not os.path.exists(path):
        return f"Error: Dataset file '{path}' not found."

    try:
        df = pd.read_csv(path)
        initial_nulls = dict(df.isnull().sum())

        # Fill numeric nulls with median
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # Fill categorical/text nulls with mode or 'Unknown'
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)

        df.to_csv(path, index=False)

        summary = f"Dataset cleaned and saved at '{path}'.\n"
        summary += f"Initial Nulls: {initial_nulls}\n"
        summary += f"Post-Cleaning Nulls: {dict(df.isnull().sum())}\n"
        summary += f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns\n"
        summary += f"Columns: {list(df.columns)}\n"
        return summary
    except Exception as e:
        return f"Error during data cleaning: {e}"


class CleanerAgents:
    def make_cleaner_agent(self, llm):
        return Agent(
            role='Expert Data Cleaner',
            goal='Use the Data Cleaner Tool to sanitize dataset on disk, handling missing values and data types efficiently.',
            backstory='A data quality specialist who uses automated tools to clean datasets and ensure data integrity.',
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[data_cleaner_tool]
        )

    def make_cleaning_task(self, agent, context):
        return Task(
            description=(
                'Use the Data Cleaner Tool to clean the dataset at "data/cleaned_dataset.csv".\n'
                'Summarize the cleaning steps taken (missing value imputation, column types) and output the list of cleaned columns.'
            ),
            expected_output='A summary of data cleaning steps and cleaned dataset metadata.',
            agent=agent,
            context=context
        )