  # main.py

from crewai import Crew, Process, LLM
from agents.cleaner_agent import CleanerAgents
from agents.data_fetcher_agent import DataFetcherAgents
from agents.visualizer_agent import VisualizerAgents
from agents.report_agent import ReportAgents
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


def run_crew(filepath: str):
    """
    Initializes and runs the data scientist agent crew.
    - filepath: The path to the CSV file to be analyzed.
    """
    # Load environment variables
    load_dotenv()

    # --- LLM Setup ---
    llm = LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0.7,
    )

    # --- Instantiate Agent Classes ---
    fetcher_agents = DataFetcherAgents()
    cleaner_agents = CleanerAgents()
    visualizer_agents = VisualizerAgents()
    report_agents = ReportAgents()

    # --- Create Individual Agents ---
    fetcher_agent = fetcher_agents.make_data_fetcher_agent(llm)
    cleaner_agent = cleaner_agents.make_cleaner_agent(llm)
    visualizer_agent = visualizer_agents.make_visualizer_agent(llm)
    report_agent = report_agents.make_report_agent(llm)

    # --- Create Tasks ---
    fetch_task = fetcher_agents.make_fetch_task(fetcher_agent, filepath)
    cleaning_task = cleaner_agents.make_cleaning_task(agent=cleaner_agent, context=[fetch_task])
    eda_task = visualizer_agents.make_eda_task(agent=visualizer_agent, context=[cleaning_task])
    report_task = report_agents.make_report_task(agent=report_agent, context=[eda_task])

    # --- Assemble Crew ---
    crew = Crew(
        agents=[fetcher_agent, cleaner_agent, visualizer_agent, report_agent],
        tasks=[fetch_task, cleaning_task, eda_task, report_task],
        process=Process.sequential,
        verbose=True
    )

    # --- Execute Workflow ---
    result = crew.kickoff()
    return result
