"""
questionEvaluationPipeline.py

author: Moneera Alsuwailm


This script evaluates the responses of a question-answering pipeline against a dataset of pre-answered questions. It leverages several NLP
metrics to assess the quality of the generated answers, such as semantic similarity, BLEU, ROUGE, and BERTScore. The script outputs a CSV file
with the evaluations and calculates aggregate statistics on the performance of the model.

Dependencies:
  - pandas
  - sklearn
  - sentence_transformers
  - nltk
  - rouge_score
  - bert_score

Usage:
  python questionEvaluationPipeline.py path/to/labeled_test_data
"""

import csv
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score
import argparse
import nltk

nltk.download('punkt')

from questionAnsweringPipeline import QuestionAnsweringPipeline


class QuestionEvaluator:
    """Class to encapsulate the evaluation of a question-answering pipeline."""

    def __init__(self, qa_pipeline, input_csv, output_dir):
        self.qa_pipeline = qa_pipeline
        self.input_csv = input_csv
        self.output_dir = output_dir

    @staticmethod
    def calculate_similarity(expected, generated):
        """Calculates the semantic similarity between two sentences."""
        model = SentenceTransformer('all-MiniLM-L12-v2')
        embeddings = model.encode([expected, generated])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def get_new_filename(self, filename):
        """Generates a unique filename to avoid overwriting existing files."""
        counter = 1
        new_filename = filename
        while os.path.exists(f"{self.output_dir}{new_filename}.csv"):
            new_filename = f"{filename}_{counter}"
            counter += 1
        return f"{new_filename}.csv"

    def evaluate(self):
        """Evaluates the question-answering pipeline and writes the results to a CSV file."""
        df = pd.read_csv(self.input_csv)

        unique_filename = self.get_new_filename('result_dataset')
        output_csv_path = os.path.join(self.output_dir, unique_filename)

        with open(output_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Prompt', 'Correct Answer', 'Generated Answer', 'Semantic Similarity', 'BLEU Score', 'ROUGE-L Score',
                 'BERTScore F1'])

            for index, row in df.iterrows():
                try:
                    generated_answer = self.qa_pipeline.answer_question(row['prompt'])['result']
                    expected_answer = row['answer']

                    similarity = self.calculate_similarity(expected_answer, generated_answer)
                    reference = word_tokenize(expected_answer.lower())
                    candidate = word_tokenize(generated_answer.lower())
                    bleu_score = sentence_bleu([reference], candidate)
                    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                    rouge_score = scorer.score(expected_answer, generated_answer)['rougeL'].fmeasure
                    _, _, bertscore = score([generated_answer], [expected_answer], lang='en',
                                            rescale_with_baseline=True)
                    bertscore_f1 = bertscore.mean().item()

                    writer.writerow(
                        [row['prompt'], expected_answer, generated_answer, similarity, bleu_score, rouge_score,
                         bertscore_f1])

                except Exception as e:
                    print(f"An error occurred while processing row {index}: {e}")

        print(f"Results written to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Question Answering Pipeline")
    parser.add_argument('input_csv', type=str, help="Input CSV file with manual Q&A pairs")
    args = parser.parse_args()

    try:
        qa_pipeline = QuestionAnsweringPipeline(
            model_path='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
            embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
            vectorstore_path='vectorstore/db_faiss'
        )

        evaluator = QuestionEvaluator(
            qa_pipeline=qa_pipeline,
            input_csv=args.input_csv,
            output_dir=os.path.join(os.getcwd(), 'data/')
        )

        evaluator.evaluate()

    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
