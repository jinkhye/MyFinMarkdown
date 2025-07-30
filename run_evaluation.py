# run_evaluation.py

import os
import pandas as pd
from langchain_openai import ChatOpenAI
from src.evaluation.llm_judge import run_llm_judge_batch
from src.evaluation.markdown_teds import MarkdownTableSimilarity

def main():
    """
    Main function to orchestrate the complete evaluation pipeline, running both
    the LLM-as-a-Judge and the Markdown TEDS metric, then combining the results.
    """
    # --- 1. Configuration ---
    # IMPORTANT: Set your OpenAI API key in your environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    # Initialize the LLM for the judge. Change the model name as needed.
    llm = ChatOpenAI(model="o3-mini", disabled_params={"parallel_tool_calls": None})

    # Define file paths for the data and results
    expected_csv_path = "examples/example_gold_output.csv"
    actual_csv_path = "examples/example_pred_output.csv"
    output_results_path = "output.csv"

    # --- 2. Load Data ---
    print("Loading evaluation data...")
    try:
        df_expected = pd.read_csv(expected_csv_path)
        df_actual = pd.read_csv(actual_csv_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        return

    if len(df_expected) != len(df_actual):
        print("Error: The 'expected' and 'actual' CSV files have a different number of rows.")
        return

    # Extract the markdown content into lists
    gold_markdowns = df_expected['output'].tolist()
    predicted_markdowns = df_actual['output'].tolist()

    # --- 3. Run Both Evaluations ---
    
    # a) Run LLM-as-a-Judge
    llm_judge_results = run_llm_judge_batch(
        gold_list=gold_markdowns,
        predicted_list=predicted_markdowns,
        llm=llm
    )
    
    # b) Run Markdown TEDS Metric
    print("\nStarting Markdown TEDS evaluation...")
    teds_calculator = MarkdownTableSimilarity()
    teds_scores = [
        teds_calculator.evaluate(pred_md=pred, true_md=gold)
        for pred, gold in zip(predicted_markdowns, gold_markdowns)
    ]
    print("Markdown TEDS evaluation complete.")

    # --- 4. Combine, Process, and Display Results ---

    # Combine all results into a single DataFrame for analysis and saving
    results_df = pd.DataFrame(llm_judge_results)
    results_df['teds_score'] = teds_scores
    
    print("\nCombined Evaluation Summary:")
    print("==================================================")
    
    # Calculate and print LLM Judge accuracy summary
    total_accuracy = 0
    criteria_columns = [col for col in results_df.columns if col not in ['teds_score']]
    
    if criteria_columns:
        print("--- LLM-as-a-Judge Results ---")
        for criterion in criteria_columns:
            # .fillna(False) handles cases where a batch might have failed
            accuracy = results_df[criterion].fillna(False).mean() * 100
            print(f"{criterion:<35}: {accuracy:.2f}%")
            total_accuracy += accuracy
        
        overall_accuracy = total_accuracy / len(criteria_columns)
        print("-" * 50)
        print(f"{'Overall LLM Judge Accuracy':<35}: {overall_accuracy:.2f}%")
        print("-" * 50)

    # Calculate and print TEDS score summary
    average_teds = results_df['teds_score'].mean() * 100
    print("--- Markdown TEDS Results ---")
    print(f"{'Average Markdown TEDS Score':<35}: {average_teds:.2f}%")
    print("==================================================")
    
    # --- 5. Save Combined Results to CSV ---
    try:
        results_df.index.name = "row_index"
        results_df.to_csv(output_results_path)
        print(f"\nCombined detailed results saved to '{output_results_path}'")
    except Exception as e:
        print(f"\nError saving combined results: {e}")

if __name__ == "__main__":
    main()