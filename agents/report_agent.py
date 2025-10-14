# agents/report_agent.py

from crewai import Agent, Task
from crewai.tools import tool
import os

# --- Define a Custom Tool to Save Reports ---
@tool("File Write Tool")
def file_write_tool(filename: str, content: str) -> str:
    """Writes the given content to a file."""
    # Ensure the 'reports' directory exists
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    filepath = os.path.join('reports', filename)
    with open(filepath, 'w') as f:
        f.write(content)
    return f"Report saved successfully as {filepath}"

class ReportAgents:
    def make_report_agent(self, llm):
        return Agent(
            role='Expert Data Science Reporter',
            goal='Generate a comprehensive and human-readable report from the data analysis and visualizations.',
            backstory='A skilled writer who specializes in translating complex data findings into clear, concise, and actionable reports.',
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[file_write_tool]
        )

    def make_report_task(self, agent, context):
        return Task(
            description=(
                'Combine all the provided information into a single, well-structured Markdown report. '
                'The report should include:\n'
                '1. An introduction summarizing the purpose of the analysis.\n'
                '2. A summary of the data cleaning process.\n'
                '3. Key insights derived from the visualizations, referencing the plot filenames.\n'
                '4. A concluding summary of the findings.\n'
                'Use the File Write Tool to save the final report as "final_report.md".'
            ),
            expected_output='A confirmation message that the report file has been saved.',
            agent=agent,
            context=context
        )