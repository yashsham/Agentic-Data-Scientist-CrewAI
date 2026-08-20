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
            goal='Synthesize the dataset summary, cleaning actions, and visual insights into a comprehensive Markdown report.',
            backstory='A skilled writer who translates complex data analysis, cleaning metrics, and chart findings into clear, actionable reports.',
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[file_write_tool]
        )

    def make_report_task(self, agent, context):
        return Task(
            description=(
                'Analyze the provided context from all previous tasks (data fetcher, cleaner, and visualizer).\n'
                'Write a detailed, professional Markdown report based strictly on the uploaded dataset.\n'
                'The report MUST include:\n'
                '1. **Executive Summary & Purpose of Analysis**: Overview of the analyzed dataset.\n'
                '2. **Data Cleaning Summary**: Detail exact null value handling, data type corrections, and transformations performed.\n'
                '3. **Key Insights & Visualization Breakdown**: Detail specific findings from each generated chart, referencing exact filenames (e.g. barplot_...png, scatterplot_...png).\n'
                '4. **Conclusion & Actionable Next Steps**: Core business insights and recommendations.\n'
                'DO NOT produce placeholder text stating data is missing. Use the full context provided.\n'
                'Use the File Write Tool to save the report as "final_report.md".'
            ),
            expected_output='Confirmation message that final_report.md has been saved successfully.',
            agent=agent,
            context=context
        )