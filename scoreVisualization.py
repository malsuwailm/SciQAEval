"""
score_visualization.py

Author: Moneera Alsuwailm


This script visualizes the evaluation scores from a question-answering pipeline. It reads a CSV file containing evaluation metrics,
calculates average scores, and generates visualizations for the distribution of Semantic Similarity, BLEU Score, ROUGE-L Score,
and BERTScore F1. The script outputs histograms and boxplots for each metric for a comprehensive understanding of the model's performance.

Dependencies:
  - pandas
  - matplotlib
  - seaborn

Usage:
  Run the script in a Python environment where the dependencies are installed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ScoreVisualizer:
    """Class to encapsulate the visualization of NLP evaluation scores."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.average_scores = {}

    def load_data(self):
        """Loads the evaluation scores from a CSV file."""
        try:
            self.data = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"The file {self.csv_path} was not found.")
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")

    def calculate_averages(self):
        """Calculates the average of each evaluation metric."""
        if self.data is not None:
            self.average_scores = {
                'Average Semantic Similarity': self.data['Semantic Similarity'].mean(),
                'Average BLEU Score': self.data['BLEU Score'].mean(),
                'Average ROUGE-L Score': self.data['ROUGE-L Score'].mean(),
                'Average BERTScore F1': self.data['BERTScore F1'].mean()
            }
            for metric, value in self.average_scores.items():
                print(f"{metric}: {value:.3f}")

    def plot_histograms(self):
        """Plots histograms for each evaluation score."""
        if self.data is not None:
            plt.figure(figsize=(14, 7))
            metrics = ['Semantic Similarity', 'BLEU Score', 'ROUGE-L Score', 'BERTScore F1']
            for i, metric in enumerate(metrics, 1):
                plt.subplot(2, 2, i)
                sns.histplot(self.data[metric], bins=20, kde=True)
                plt.title(f'{metric} Distribution')
            plt.tight_layout()
            plt.show()

    def plot_boxplots(self):
        """Plots boxplots to compare the evaluation scores."""
        if self.data is not None:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.data[['Semantic Similarity', 'BLEU Score', 'ROUGE-L Score', 'BERTScore F1']])
            plt.title('Score Comparison Boxplot')
            plt.ylabel('Scores')
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    csv_file_path = ''
    visualizer = ScoreVisualizer(csv_file_path)
    visualizer.load_data()
    visualizer.calculate_averages()
    visualizer.plot_histograms()
    visualizer.plot_boxplots()
