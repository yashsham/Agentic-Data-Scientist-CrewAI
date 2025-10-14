# agents/cleaner_agent.py

from crewai import Agent, Task

# agents/cleaner_agent.py

# ... imports

class CleanerAgents:
    # Add 'llm' as an argument here
    def make_cleaner_agent(self, llm):
        return Agent(
            role='Expert Data Cleaner',
            goal='Analyze the provided DataFrame...',
            backstory='A meticulous data cleaner...',
            verbose=True,
            allow_delegation=False,
            llm=llm # And pass it to the agent here
        )
    
    # ... (the make_cleaning_task method stays the same) ...

    def make_cleaning_task(self, agent, context):
        return Task(
            description=(
                'Take the DataFrame and perform the following cleaning actions:\n'
                '1. Check for any missing values (nulls) in the dataset.\n'
                '2. If missing values are found, fill them with a reasonable default (e.g., 0 for numbers, "N/A" for text).\n'
                '3. Check the data type of each column and correct them if necessary (e.g., ensure Price is a number).\n'
                '4. Return the cleaned DataFrame.'
            ),
            expected_output='A cleaned pandas DataFrame with no missing values and corrected data types.',
            agent=agent,
            context=context # This is how we pass data from the previous task
        )