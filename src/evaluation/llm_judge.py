# src/evaluation/llm_judge.py

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, List

# 1. Define the structured output schema for the LLM response.
class EvaluationResponse(BaseModel):
    """Defines the structure for the evaluation criteria returned by the LLM."""
    criteria: Dict[str, bool] = Field(
        description="A dictionary of evaluation criteria and their boolean outcomes."
    )

# 2. Define the prompt template function. This is a helper for the main function.
def create_prompt(gold: str, predicted: str) -> str:
    return f"""
You are an expert Markdown evaluator for financial statement pages. Assess the 'Actual Output' against the 'Expected Output' based on the rules below.

Return:
- "criteria": A dictionary with keys for each criterion and boolean values (True if met, False if not).

---

**Evaluation Criteria for Markdown Documents:**

1. **Correct Row Count**: All tables in the Actual Output have the same number of rows as in the Expected Output.
2. **Correct Column Count**: All tables in the Actual Output have the same number of columns as in the Expected Output.
3. **Semantically Accurate Headers**: All table headers in the Actual Output convey the same meaning as those in the Expected Output (minor wording differences are acceptable if the intent is preserved).
4. **Correct Item Order**: All table items and cell values in the Actual Output maintain the same order as in the Expected Output without shifts or misplacements.
5. **Valid Markdown Formatting**: The Actual Output uses correct Markdown syntax (e.g., proper table structure, header syntax) consistent with the Expected Output.

---

Test Case:
Actual Output:
{predicted}

Expected Output:
{gold}

---
**Example Response (LLM must follow this strict format):**
{{
    "criteria": {{
        "Correct Row Count": false,
        "Correct Column Count": true,
        "Semantically Accurate Headers": false,
        "Correct Item Order": true,
        "Valid Markdown Formatting": true,
    }}
}}
"""

# 3. This is the main function that `run_evaluation.py` will import and use.
def run_llm_judge_batch(
    gold_list: List[str],
    predicted_list: List[str],
    llm: ChatOpenAI,
    batch_size: int = 10
) -> List[Dict[str, bool]]:
    """
    Evaluates markdown outputs in batches using an LLM judge.

    This function is designed to be imported and used by other scripts.

    Args:
        gold_list (List[str]): A list of the ground truth markdown strings.
        predicted_list (List[str]): A list of the predicted markdown strings.
        llm (ChatOpenAI): An initialized instance of the ChatOpenAI model.
        batch_size (int): The number of evaluations to process per API call.

    Returns:
        List[Dict[str, bool]]: A list of criteria dictionaries for each evaluation.
    """
    if len(gold_list) != len(predicted_list):
        raise ValueError("Gold and predicted lists must have the same length.")

    # Prepare messages for batch processing
    print("Preparing prompts for the LLM Judge...")
    prompts = [create_prompt(gold, pred) for gold, pred in zip(gold_list, predicted_list)]
    batch_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

    structured_llm = llm.with_structured_output(EvaluationResponse, method="function_calling")
    
    results = []
    print(f"Starting LLM Judge evaluation for {len(batch_messages)} items...")
    for i in range(0, len(batch_messages), batch_size):
        batch = batch_messages[i:i + batch_size]
        try:
            batch_results = structured_llm.batch(batch)
            # Use .model_dump() to get a dictionary, then access the 'criteria' key
            results.extend([res.model_dump()['criteria'] for res in batch_results])
            print(f"Processed batch {i//batch_size + 1}/{(len(batch_messages) + batch_size - 1) // batch_size}")
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Append empty dictionaries to maintain alignment if a batch fails
            results.extend([{}] * len(batch))
            
    print("LLM Judge evaluation complete.")
    return results