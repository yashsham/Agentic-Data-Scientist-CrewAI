# agents/visualizer_agent.py

from crewai import Agent, Task
from crewai.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def auto_create_plots(csv_path: str = "data/cleaned_dataset.csv") -> list:
    """
    Programmatically inspects the dataset at csv_path (or fallback locations) and generates 
    a suite of 4 high-quality, publication-grade visualizations in reports/ directory.
    Optimized to handle large datasets (e.g. 37.8MB+) lightning fast.
    """
    if not os.path.exists('reports'):
        os.makedirs('reports', exist_ok=True)

    path = str(csv_path).strip().strip("'").strip('"')
    
    # Fallback path checking
    if not os.path.exists(path):
        path = os.path.join("data", "cleaned_dataset.csv")
    if not os.path.exists(path) and os.path.exists("data"):
        csv_files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".csv")]
        if csv_files:
            path = csv_files[0]

    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    if df.empty:
        return []

    generated_files = []
    sns.set_theme(style="whitegrid")

    # Fast sampling for high-density plots on huge datasets (>50k rows)
    plot_df = df.sample(n=50000, random_state=42) if len(df) > 50000 else df

    num_cols = [c for c in df.select_dtypes(include=['number']).columns if 'id' not in c.lower()]
    if not num_cols:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()

    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if 'id' not in c.lower()]
    if not cat_cols:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 1. Bar Plot (Categorical vs Numeric)
    if cat_cols and num_cols:
        cat_var = cat_cols[0]
        num_var = num_cols[0]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        if df[cat_var].nunique() > 8:
            top_data = df.groupby(cat_var)[num_var].mean().nlargest(10).reset_index()
            sns.barplot(data=top_data, y=cat_var, x=num_var, hue=cat_var, legend=False, ax=ax, palette="Blues_r")
            ax.set_ylabel(cat_var.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_xlabel(f"Average {num_var.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
        else:
            top_data = df.groupby(cat_var)[num_var].mean().reset_index()
            sns.barplot(data=top_data, x=cat_var, y=num_var, hue=cat_var, legend=False, ax=ax, palette="Blues_r")
            ax.set_xlabel(cat_var.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_ylabel(f"Average {num_var.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
        
        ax.set_title(f"Average {num_var.replace('_', ' ').title()} by {cat_var.replace('_', ' ').title()}", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        filename = f"reports/barplot_{cat_var}_vs_{num_var}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        generated_files.append(filename)

    # 2. Scatter Plot (Numeric vs Numeric)
    if len(num_cols) >= 2:
        x_num = num_cols[0]
        y_num = num_cols[1]
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        sns.scatterplot(data=plot_df, x=x_num, y=y_num, ax=ax, color='#1f77b4', s=60, alpha=0.7)
        ax.set_xlabel(x_num.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel(y_num.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_title(f"{x_num.replace('_', ' ').title()} vs {y_num.replace('_', ' ').title()}", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        filename = f"reports/scatterplot_{x_num}_vs_{y_num}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        generated_files.append(filename)

    # 3. Distribution Histogram
    if num_cols:
        target_num = num_cols[0]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        sns.histplot(data=plot_df, x=target_num, kde=True, ax=ax, color='#1f77b4')
        ax.set_xlabel(target_num.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
        ax.set_title(f"Distribution of {target_num.replace('_', ' ').title()}", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        filename = f"reports/hist_{target_num}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        generated_files.append(filename)

    # 4. Correlation Heatmap
    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix Heatmap", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        filename = "reports/heatmap_correlation.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        generated_files.append(filename)

    return generated_files


@tool("Plotting Tool")
def plotting_tool(plot_type: str = "auto", x: str = "", y: str = "", csv_path: str = "data/cleaned_dataset.csv", title: str = "") -> str:
    """
    Creates and saves high-quality publication-grade visualizations from the dataset on disk.
    If plot_type is 'auto' or unspecified, it automatically generates all relevant charts.
    """
    path = str(csv_path).strip().strip("'").strip('"')
    files = auto_create_plots(path)
    if files:
        return f"Visualizations generated successfully! Saved charts: {', '.join(files)}"

    return "No visualization files generated."


class VisualizerAgents:
    def make_visualizer_agent(self, llm):
        return Agent(
            role='Expert Data Visualizer',
            goal='Execute the Plotting Tool to generate publication-grade charts from "data/cleaned_dataset.csv".',
            backstory='A data visualization specialist who invokes plotting tools to save charts to disk.',
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[plotting_tool]
        )

    def make_eda_task(self, agent, context):
        return Task(
            description=(
                'Invoke the Plotting Tool with plot_type="auto" to automatically generate all required visualizations from "data/cleaned_dataset.csv".\n'
                'Report the exact list of generated chart image filenames in your final output.'
            ),
            expected_output='A summary confirming generated chart image filenames in the reports/ directory.',
            agent=agent,
            context=context
        )