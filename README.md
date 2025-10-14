# Agentic Data Scientist 🤖

This project implements an autonomous AI "team" of specialized agents that collaborate to perform a complete data science workflow. This system, built with CrewAI, demonstrates a sophisticated multi-agent architecture capable of handling data ingestion, cleaning, exploratory data analysis (EDA), and final report generation.

## The Agent Crew 🧑‍💻

The system is composed of four specialized agents:

1.  **🧩 DataFetcherAgent:** Ingests datasets from local files (e.g., CSV).
2.  **🧼 CleanerAgent:** Analyzes the data for inconsistencies, handles missing values, and corrects data types.
3.  **📊 VisualizerAgent:** Performs EDA, generates insightful plots (bar charts, scatter plots), and saves them as image files.
4.  **🗒️ ReportAgent:** Synthesizes all findings and visualizations into a comprehensive, human-readable Markdown report.

## Tech Stack 🛠️

* **Agent Framework:** CrewAI
* **LLM:** Google Gemini Pro
* **Core Libraries:** LangChain, Pandas, Matplotlib, Seaborn
* **UI/Reporting:** Streamlit (for potential future interface)

## Setup & Installation ⚙️

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/Agentic-Data-Scientist-CrewAI.git](https://github.com/YOUR_USERNAME/Agentic-Data-Scientist-CrewAI.git)
cd Agentic-Data-Scientist-CrewAI
```

**2. Create and activate a virtual environment:**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up your API Key:**
Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

## How to Run 🚀

Execute the main script from your terminal to kick off the agent crew:

```bash
python main.py
```

The process will run sequentially, and you can find the generated plots and the final Markdown report inside the `reports/` directory.