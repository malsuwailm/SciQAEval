# SciQAEval

SciQAEval evaluates the performance of question-answering systems against scientific publications. It analyzes answers generated by a QA model, computes key NLP metrics, and generates visualizations for a clear understanding of the model's capabilities.

## Features

- **Question Evaluation**: Leverage metrics like semantic similarity, BLEU, ROUGE, and BERTScore to evaluate the quality of answers.
- **Score Visualization**: Visualize the distribution of scores and compare the performance across different metrics with histograms and boxplots.
- **Average Score Calculation**: Compute and output the average for each metric for a quick assessment of overall performance.

## Dependencies

Ensure you have the following Python libraries installed:

- pandas
- matplotlib
- seaborn
- scikit-learn
- sentence_transformers
- nltk
- rouge_score
- bert_score

## Installation

Clone this repository to your local machine using:

```
git clone https://github.com/malsuwailm/SciQAEval.git
```

Install the required dependencies:

```
pip install -r requirements.txt
```

## Setting up PyTorch with CUDA Support

To leverage GPU acceleration with PyTorch, ensure you have a CUDA-compatible GPU and the appropriate version of CUDA installed. Follow these steps to install PyTorch with CUDA support:

1. First, make sure you have the NVIDIA CUDA Toolkit installed. The version of the toolkit should be compatible with the PyTorch build you intend to install.

2. Use the following command to install PyTorch with CUDA support through Conda:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

This command installs PyTorch with CUDA 11.8 support, which allows you to run PyTorch on the GPU.

1. Verify the installation by running:

```
import torch
print(torch.cuda.is_available())  # This should return True if CUDA is properly set up
```

## Usage

Run the questionAnsweringPipeline.py script with your question regarding the scientific article:

```
python questionAnsweringPipeline.py "your question here"
```

Run the questionEvaluationPipeline.py script with the path to your labeled test data in CSV format:

```
python questionEvaluationPipeline.py path/to/labeled_test_data.csv
```

After running the evaluation pipeline, you can visualize the scores using scoreVisualization.py:

```
python scoreVisualization.py
```

## Model Performance

## Average Evaluation Scores

| Metric                    | Average Score | Description |
|---------------------------|---------------|-------------|
| Semantic Similarity       | 0.713         | Indicates that the generated answers are, on average, semantically similar to the correct answers. |
| BLEU Score                | 0.103         | Suggests that answers may be correct but phrased differently than the references. |
| ROUGE-L Score             | 0.305         | Implies a moderate structural similarity between the generated and reference answers. |
| BERTScore F1              | 0.181         | On the lower side, highlighting potential areas for the model to improve in generating semantically precise answers. |

\
&nbsp;

![Histograms](https://github.com/malsuwailm/SciQAEval/blob/main/visualizations/bars.png)


## Contributing

Contributions to SciQAEval are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

Distributed under the MIT License. See LICENSE for more information.

