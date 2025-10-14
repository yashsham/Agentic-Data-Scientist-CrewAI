# agents/data_fetcher_agent.py

from crewai import Agent, Task
import pandas as pd

# agents/data_fetcher_agent.py

# ... imports

class DataFetcherAgents:
    # Add 'llm' as an argument here
    def make_data_fetcher_agent(self, llm): 
        return Agent(
            role='Expert Data Fetcher',
            goal='Fetch a dataset from a specified file path and load it into a pandas DataFrame.',
            backstory='An expert in data ingestion...',
            verbose=True,
            allow_delegation=False,
            llm=llm # And pass it to the agent here
        )
    

    # Add this new method
    def make_fetch_task(self, agent, file_path):
        return Task(
            description=f'Fetch the dataset from the file path {file_path}.', # The specific instruction
            expected_output='A pandas DataFrame containing the data from the file.', # What success looks like
            agent=agent # The agent assigned to this task
        )