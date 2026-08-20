# agents/report_agent.py

from crewai import Agent, Task
from crewai.tools import tool
import os


@tool("File Write Tool")
def file_write_tool(filename: str, content: str) -> str:
    """Writes the given content to a file in the reports directory."""
    if not os.path.exists('reports'):
        os.makedirs('reports', exist_ok=True)

    filepath = os.path.join('reports', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Report saved successfully as {filepath}"


class ReportAgents:
    def make_report_agent(self, llm):
        return Agent(
            role='Expert Data Science Reporter',
            goal='Generate an accurate, comprehensive Markdown report strictly reflecting the analyzed dataset columns and generated visualizations.',
            backstory='A meticulous data science communicator who synthesizes data findings into actionable, professional reports.',
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[file_write_tool]
        )

    def make_report_task(self, agent, context):
        return Task(
            description=(
                'Combine all information from previous tasks into a well-structured Markdown report strictly based on the uploaded dataset.\n'
                'The report MUST include:\n'
                '1. Executive Summary & Purpose of Analysis.\n'
                '2. Data Cleaning Summary (null value handling, data type conversions, scaling/filtering).\n'
                '3. Key Insights & Visualization Breakdown (reference exact generated plot filenames like barplot_...png or scatterplot_...png).\n'
                '4. Actionable Conclusion & Recommendations.\n'
                'Save the final Markdown report as "final_report.md" using the File Write Tool.'
            ),
            expected_output='A confirmation message that final_report.md has been saved successfully.',
            agent=agent,
            context=context
        )