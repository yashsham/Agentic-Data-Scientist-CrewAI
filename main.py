# main.py

import os
import glob
import logging

# Disable CrewAI telemetry and interactive trace prompt stalls
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from dotenv import load_dotenv
from crewai import Crew, Process
from agents.cleaner_agent import CleanerAgents
from agents.data_fetcher_agent import DataFetcherAgents
from agents.visualizer_agent import VisualizerAgents, auto_create_plots
from agents.report_agent import ReportAgents
from utils.llm_manager import get_automatic_fallback_chain

logger = logging.getLogger(__name__)


def clean_reports_directory():
    """Removes old reports and PNG plot images before kicking off a new run."""
    if os.path.exists('reports'):
        for file in glob.glob("reports/*"):
            try:
                if os.path.isfile(file):
                    os.remove(file)
            except Exception as e:
                logger.warning(f"Could not remove old file {file}: {e}")
    else:
        os.makedirs('reports', exist_ok=True)


def clean_report_markdown(raw_text: str) -> str:
    """Strips residual ReAct thought prefixes to output clean Markdown text."""
    if not raw_text:
        return "# Analysis Report\n\nAnalysis completed successfully."
    
    text = str(raw_text).strip()
    
    # If text contains markdown header, cut off prior thinking monologue
    if "## 1. Executive Summary" in text:
        idx = text.find("## 1. Executive Summary")
        # Check if there is an H1 header right before
        h1_idx = text.rfind("# ", 0, idx)
        if h1_idx != -1:
            text = text[h1_idx:]
        else:
            text = text[idx:]
    elif "# " in text:
        idx = text.find("# ")
        text = text[idx:]
        
    return text


def run_crew(filepath: str, status_callback=None):
    """
    Initializes and runs the data scientist agent crew with automatic backend fallback support.
    - filepath: The path to the CSV file to be analyzed.
    - status_callback: Optional status logger callback for UI alerts.
    """
    load_dotenv()

    # Clean out previous run outputs so old plots/reports are not cached
    clean_reports_directory()

    # Automatically construct fallback chain from backend environment variables/secrets
    candidates = get_automatic_fallback_chain()

    last_exception = None
    for idx, (provider, model_name, selected_llm) in enumerate(candidates):
        msg = f"Attempting execution using {provider.upper()} ({model_name})..."
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

            # Create Tasks with complete sequential context
            fetch_task = fetcher_agents.make_fetch_task(fetcher_agent, filepath)
            cleaning_task = cleaner_agents.make_cleaning_task(agent=cleaner_agent, context=[fetch_task])
            eda_task = visualizer_agents.make_eda_task(agent=visualizer_agent, context=[cleaning_task])
            report_task = report_agents.make_report_task(agent=report_agent, context=[fetch_task, cleaning_task, eda_task])

            # Assemble Crew
            crew = Crew(
                agents=[fetcher_agent, cleaner_agent, visualizer_agent, report_agent],
                tasks=[fetch_task, cleaning_task, eda_task, report_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute Workflow
            result = crew.kickoff()

            # Ensure plots exist on disk even if agent tool call returned text only
            if not glob.glob("reports/*.png"):
                auto_create_plots("data/cleaned_dataset.csv")

            # Always save cleaned Markdown report to reports/final_report.md
            raw_output = result.raw if hasattr(result, 'raw') else str(result)
            cleaned_report = clean_report_markdown(raw_output)
            
            os.makedirs("reports", exist_ok=True)
            report_path = os.path.join("reports", "final_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(cleaned_report)

            success_msg = f"Crew completed successfully with {provider.upper()} ({model_name})!"
            logger.info(success_msg)
            if status_callback:
                status_callback({
                    "status": "success",
                    "provider": provider,
                    "model": model_name,
                    "message": success_msg
                })

            return result

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
                fallback_msg = f"Primary provider failed. Falling back to backend provider {next_provider.upper()}..."
                logger.info(fallback_msg)
                if status_callback:
                    status_callback({
                        "status": "fallback",
                        "next_provider": next_provider,
                        "message": fallback_msg
                    })

    raise RuntimeError(f"All LLM providers in fallback chain failed. Last error: {last_exception}")
