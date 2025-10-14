# agents/visualizer_agent.py

# NEW: Import the 'tool' decorator
from crewai import Agent, Task
from crewai.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# REWRITTEN: The class is now a simple function with a decorator
@tool("Plotting Tool")
def plotting_tool(df_json: str, plot_type: str, x: str, y: str = None) -> str:
    """
    Creates and saves a plot from a JSON representation of a DataFrame.
    - df_json: The DataFrame in JSON format.
    - plot_type: The type of plot to create (e.g., 'bar', 'scatter').
    - x: The column for the x-axis.
    - y: The column for the y-axis (optional for some plots).
    """
    if not os.path.exists('reports'):
        os.makedirs('reports')

    df = pd.read_json(df_json)
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'bar':
        sns.barplot(data=df, x=x, y=y)
        filename = f"reports/barplot_{x}_vs_{y}.png"
    elif plot_type == 'scatter':
        sns.scatterplot(data=df, x=x, y=y)
        filename = f"reports/scatterplot_{x}_vs_{y}.png"
    else:
        return "Unsupported plot type."

    plt.savefig(filename)
    plt.close()
    return f"Plot saved successfully as {filename}"


class VisualizerAgents:
    def make_visualizer_agent(self, llm):
        return Agent(
            role='Expert Data Visualizer',
            goal='Create insightful visualizations from the provided dataset to uncover trends, patterns, and outliers. Use the provided tools to save plots.',
            backstory='An acclaimed data artist who can turn numbers into compelling stories through beautiful and informative charts.',
            verbose=True,
            allow_delegation=False,
            llm=llm,
            # Pass the function directly into the tools list
            tools=[plotting_tool] 
        )

    def make_eda_task(self, agent, context):
        return Task(
            description=(
                'Analyze the cleaned DataFrame to identify key relationships between columns. '
                'Then, use the Plotting Tool to create and save AT LEAST TWO different types of visualizations (e.g., a bar plot and a scatter plot) that reveal important insights. '
                'Focus on the relationship between "Price" and other variables.'
            ),
            expected_output='A confirmation message for each plot saved, indicating the filenames.',
            agent=agent,
            context=context
        )