# agents/report_agent.py

from crewai import Agent, Task


class ReportAgents:
    def make_report_agent(self, llm):
        return Agent(
            role='Expert Data Science Reporter',
            goal='Synthesize the dataset summary, data cleaning actions, and visual chart insights into a publication-grade Markdown report.',
            backstory='A senior data science editor who transforms raw analysis and chart metadata into executive-level reports.',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    def make_report_task(self, agent, context):
        return Task(
            description=(
                'Review the dataset summaries, cleaning steps, and generated chart filenames provided in the context.\n'
                'Write a comprehensive, publication-grade Markdown report for the uploaded dataset.\n'
                'The report MUST be structured with the following exact level-2 headers:\n'
                '## 1. Executive Summary & Purpose of Analysis\n'
                '## 2. Data Cleaning & Sanitization Summary\n'
                '## 3. Key Insights & Visualization Breakdown\n'
                '## 4. Conclusion & Actionable Recommendations\n\n'
                'Requirements:\n'
                '- Reference exact generated plot filenames in Section 3 (e.g. `barplot_...png`, `scatterplot_...png`, `hist_...png`, `heatmap_correlation.png`).\n'
                '- Do NOT include meta-commentary, ReAct thoughts, or instructions in your final output.\n'
                '- Output ONLY the final formatted Markdown report text.'
            ),
            expected_output='A complete, publication-grade Markdown report starting directly with the title and sections.',
            agent=agent,
            context=context
        )