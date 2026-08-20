# main.py

import logging
from dotenv import load_dotenv
from crewai import Crew, Process
from agents.cleaner_agent import CleanerAgents
from agents.data_fetcher_agent import DataFetcherAgents
from agents.visualizer_agent import VisualizerAgents
from agents.report_agent import ReportAgents
from utils.llm_manager import get_configured_llm_chain, build_llm_instance

logger = logging.getLogger(__name__)


def run_crew(filepath: str, llm=None, llm_chain=None, status_callback=None):
    """
    Initializes and runs the data scientist agent crew with fallback support.
    - filepath: The path to the CSV file to be analyzed.
    - llm: A specific LLM instance (optional).
    - llm_chain: A list of (provider, model_name, LLM_instance) tuples to attempt sequentially.
    - status_callback: Optional callback function to report status updates to UI (e.g. Streamlit).
    """
    # Load environment variables
    load_dotenv()

    # Determine candidates to run
    candidates = []
    if llm:
        model_name = getattr(llm, "model", "custom")
        candidates = [("custom", model_name, llm)]
    elif llm_chain and len(llm_chain) > 0:
        candidates = llm_chain
    else:
        candidates = get_configured_llm_chain()

    if not candidates:
        default_llm = build_llm_instance("gemini")
        candidates = [("gemini", "gemini/gemini-2.0-flash", default_llm)]

    last_exception = None
    for idx, (provider, model_name, selected_llm) in enumerate(candidates):
        msg = f"Attempting Crew execution using {provider.upper()} ({model_name})..."
        logger.info(msg)
        if status_callback:
            status_callback({
                "status": "attempting",
                "provider": provider,
                "model": model_name,
                "message": msg,
                "attempt": idx + 1
            })

        try:
            # Instantiate Agent Classes
            fetcher_agents = DataFetcherAgents()
            cleaner_agents = CleanerAgents()
            visualizer_agents = VisualizerAgents()
            report_agents = ReportAgents()

            # Create Individual Agents
            fetcher_agent = fetcher_agents.make_data_fetcher_agent(selected_llm)
            cleaner_agent = cleaner_agents.make_cleaner_agent(selected_llm)
            visualizer_agent = visualizer_agents.make_visualizer_agent(selected_llm)
            report_agent = report_agents.make_report_agent(selected_llm)

            # Create Tasks
            fetch_task = fetcher_agents.make_fetch_task(fetcher_agent, filepath)
            cleaning_task = cleaner_agents.make_cleaning_task(agent=cleaner_agent, context=[fetch_task])
            eda_task = visualizer_agents.make_eda_task(agent=visualizer_agent, context=[cleaning_task])
            report_task = report_agents.make_report_task(agent=report_agent, context=[eda_task])

            # Assemble Crew
            crew = Crew(
                agents=[fetcher_agent, cleaner_agent, visualizer_agent, report_agent],
                tasks=[fetch_task, cleaning_task, eda_task, report_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute Workflow
            result = crew.kickoff()

            success_msg = f"Crew completed successfully with {provider.upper()} ({model_name})!"
            logger.info(success_msg)
            if status_callback:
                status_callback({
                    "status": "success",
                    "provider": provider,
                    "model": model_name,
                    "message": success_msg
                })

            return {
                "result": result,
                "provider": provider,
                "model": model_name,
                "used_fallback": idx > 0,
                "attempts": idx + 1
            }

        except Exception as e:
            last_exception = e
            err_msg = f"Execution failed on {provider.upper()} ({model_name}): {str(e)}"
            logger.warning(err_msg)
            if status_callback:
                status_callback({
                    "status": "failed",
                    "provider": provider,
                    "model": model_name,
                    "error": str(e),
                    "message": err_msg
                })

            if idx < len(candidates) - 1:
                next_provider = candidates[idx + 1][0]
                fallback_msg = f"Falling back to provider {next_provider.upper()}..."
                logger.info(fallback_msg)
                if status_callback:
                    status_callback({
                        "status": "fallback",
                        "next_provider": next_provider,
                        "message": fallback_msg
                    })

    raise RuntimeError(f"All LLM providers in fallback chain failed. Last error: {last_exception}")
