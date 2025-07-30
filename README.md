# [Evaluation Tools] Fine-Tuning Vision-Language Models for Structured Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports

<img width="1563" height="637" alt="image" src="https://github.com/user-attachments/assets/c8c3897e-1b09-4054-8cd0-cdeb3e2a0200" />

## Description

This repository provides the tools for evaluating the quality of Markdown tables generated from financial documents. It includes scripts for automated, criteria-based assessment using a Large Language Model (LLM-as-a-Judge) and for calculating structural similarity using the Markdown Tree-Edit-Distance-based Similarity (TEDS) metric.

These tools were developed for the research paper: **"Fine-Tuning Vision-Language Models for Structured Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports"**.

## Tools Overview

The repository is structured around three key components:

1.  **`run_evaluation.py`**: The main entry point script that orchestrates the entire evaluation process. It loads your data, runs both the LLM Judge and TEDS evaluations, and outputs a combined summary.
2.  **`src/evaluation/llm_judge.py`**: A module containing the logic for the "LLM-as-a-Judge". It sends the ground-truth and predicted markdown to an LLM (e.g., o3-mini) and asks it to score the output based on a predefined set of quality criteria.
3.  **`src/evaluation/markdown_teds.py`**: A module implementing the **Markdown TEDS** metric. It parses markdown tables into tree structures and calculates a holistic similarity score (from 0.0 to 1.0) based on structural and content differences.

## Usage Guide

Follow these steps to set up and run the evaluation on your own data.

### Step 1: Setup the Environment

First, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/jinkhye/MyFinMarkdown.git
cd MyFinMarkdown

# Create a requirements.txt file with the content below
# and then install the dependencies.
pip install -r requirements.txt
```

### Step 2: Set Your OpenAI API Key

The LLM-as-a-Judge requires access to the OpenAI API. Set your API key as an environment variable in your terminal.

**On macOS / Linux:**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

**On Windows (Command Prompt):**
```bash
set OPENAI_API_KEY="your_api_key_here"
```

### Step 3: Prepare Your Data

Place your data files in a directory structure that the scripts expect. The evaluation requires two CSV files: one with the ground-truth ("expected") outputs and one with your model's ("actual") outputs.

1.  Ensure both CSV files have a column named **`output`** that contains the raw markdown text for each table.

2.  Open `run_evaluation.py` and update the file paths to point to your specific CSV files:
    ```python
    # Inside run_evaluation.py
    expected_csv_path = "{your_dataset_path}/{expected_output}.csv"
    actual_csv_path = "{your_dataset_path}/{actual_output}.csv"
    ```

### Step 4: Run the Evaluation

Execute the main script from your terminal. It will run both the LLM Judge and TEDS evaluations and print a summary.

```bash
python run_evaluation.py
```

### Understanding the Output

1.  **Console Output**: A summary report will be printed directly to your terminal, showing the accuracy for each LLM Judge criterion and the final average Markdown TEDS score.

2.  **CSV File**: A detailed, row-by-row report named `output.csv` will be saved in your project's root directory. This file contains the boolean result for each criterion and the TEDS score for every sample.