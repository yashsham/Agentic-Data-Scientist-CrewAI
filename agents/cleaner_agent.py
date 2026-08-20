# agents/cleaner_agent.py

from crewai import Agent, Task


class CleanerAgents:
    def make_cleaner_agent(self, llm):
        return Agent(
            role='Expert Data Cleaner',
            goal='Analyze the fetched dataset, clean missing values, correct column data types, and output the cleaned dataset in JSON format.',
            backstory='A meticulous data cleaner who transforms raw datasets into sanitized, analysis-ready DataFrames.',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    def make_cleaning_task(self, agent, context):
        return Task(
            description=(
                'Take the dataset overview from the previous task and perform data cleaning:\n'
                '1. Check for null or missing values across all columns and fill/impute them appropriately.\n'
                '2. Verify and fix data types (e.g. numeric columns converted to float/int).\n'
                '3. Output a step-by-step summary of all cleaning actions taken.\n'
                '4. Include the cleaned DataFrame in JSON format so the visualizer agent can generate charts.'
            ),
            expected_output='A summary of all data cleaning actions and the cleaned DataFrame in JSON format.',
            agent=agent,
            context=context
        )