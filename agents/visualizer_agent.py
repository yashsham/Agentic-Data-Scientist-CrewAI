# agents/visualizer_agent.py

from crewai import Agent, Task
from crewai.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io


@tool("Plotting Tool")
def plotting_tool(df_json: str, plot_type: str, x: str, y: str = None, title: str = None) -> str:
    """
    Creates and saves high-quality publication-grade visualizations from a JSON representation of a DataFrame.
    - df_json: The DataFrame in JSON format.
    - plot_type: The type of plot to create ('bar', 'scatter', 'hist', 'line', 'box').
    - x: The column name for the x-axis.
    - y: The column name for the y-axis (optional for some plots).
    - title: Custom title for the plot (optional).
    """
    if not os.path.exists('reports'):
        os.makedirs('reports', exist_ok=True)

    try:
        df = pd.read_json(io.StringIO(df_json))
    except Exception:
        df = pd.read_json(df_json)

    # Modern aesthetic theme setup
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    clean_x = str(x).strip() if x else ""
    clean_y = str(y).strip() if y and y != "None" else None

    if plot_type in ['bar', 'barplot']:
        # High cardinality handling for long categorical variables
        if clean_x in df.columns and df[clean_x].nunique() > 8:
            if clean_y and clean_y in df.columns:
                top_data = df.groupby(clean_x)[clean_y].mean().nlargest(10).reset_index()
            else:
                top_data = df[clean_x].value_counts().nlargest(10).reset_index()
                top_data.columns = [clean_x, 'count']
                clean_y = 'count'

            # Horizontal bar plot for long names to avoid overlap
            sns.barplot(data=top_data, y=clean_x, x=clean_y, hue=clean_x, legend=False, ax=ax, palette="Blues_r")
            ax.set_ylabel(clean_x.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_xlabel(f"Average {clean_y.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
        else:
            sns.barplot(data=df, x=clean_x, y=clean_y, hue=clean_x, legend=False, ax=ax, palette="Blues_r")
            ax.set_xlabel(clean_x.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            if clean_y:
                ax.set_ylabel(clean_y.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            plt.xticks(rotation=45, ha='right')

        filename = f"reports/barplot_{clean_x}_vs_{clean_y}.png"

    elif plot_type in ['scatter', 'scatterplot']:
        sns.scatterplot(data=df, x=clean_x, y=clean_y, ax=ax, color='#1f77b4', s=80, alpha=0.8)
        ax.set_xlabel(clean_x.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        if clean_y:
            ax.set_ylabel(clean_y.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        filename = f"reports/scatterplot_{clean_x}_vs_{clean_y}.png"

    elif plot_type in ['hist', 'histogram']:
        sns.histplot(data=df, x=clean_x, kde=True, ax=ax, color='#1f77b4')
        ax.set_xlabel(clean_x.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
        filename = f"reports/hist_{clean_x}.png"

    elif plot_type in ['box', 'boxplot']:
        if clean_x in df.columns and df[clean_x].nunique() > 8:
            top_cats = df[clean_x].value_counts().nlargest(8).index
            sub_df = df[df[clean_x].isin(top_cats)]
        else:
            sub_df = df
        sns.boxplot(data=sub_df, x=clean_x, y=clean_y, hue=clean_x, legend=False, ax=ax, palette="Set2")
        ax.set_xlabel(clean_x.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        if clean_y:
            ax.set_ylabel(clean_y.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        filename = f"reports/boxplot_{clean_x}_vs_{clean_y}.png"

    else:
        sns.scatterplot(data=df, x=clean_x, y=clean_y, ax=ax, color='#1f77b4')
        filename = f"reports/plot_{clean_x}.png"

    # Title formatting
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        plot_title = f"{plot_type.capitalize()} Plot of {clean_x.replace('_', ' ')}"
        if clean_y:
            plot_title += f" vs {clean_y.replace('_', ' ')}"
        ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return f"Plot saved successfully as {filename}"


class VisualizerAgents:
    def make_visualizer_agent(self, llm):
        return Agent(
            role='Expert Data Visualizer',
            goal='Create clear, publication-grade visualizations from the provided dataset to uncover key insights. Use the Plotting Tool to save charts.',
            backstory='An acclaimed data artist who transforms numbers into beautifully styled, easy-to-read charts with non-overlapping labels.',
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[plotting_tool]
        )

    def make_eda_task(self, agent, context):
        return Task(
            description=(
                'Analyze the cleaned DataFrame to identify key relationships between columns. '
                'Then, use the Plotting Tool to create and save AT LEAST TWO different types of clear, un-cluttered visualizations '
                '(e.g., a bar plot of top categories and a scatter plot or histogram) that reveal important insights. '
                'Focus on key target variables such as Price, UnitPrice, Quantity, or Category.'
            ),
            expected_output='A confirmation message for each plot saved, indicating the filenames.',
            agent=agent,
            context=context
        )