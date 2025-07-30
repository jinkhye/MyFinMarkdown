# markdown_teds.py

"""
A metric for evaluating Markdown table similarity, based on the TEDS framework.

This script can be run directly to calculate the TEDS score between two CSV files
containing 'expected' and 'actual' markdown strings.
"""

import re
import os
import pandas as pd
from typing import List, Optional

import distance
import numpy as np
from apted import APTED, Config
from apted.helpers import Tree
from lxml import html
from markdown_it import MarkdownIt
from scipy.optimize import linear_sum_assignment


class _TableTree(Tree):
    """A custom tree structure to represent a table, compatible with APTED."""

    def __init__(self, tag: str, content: str = "", *children: "_TableTree"):
        self.tag = tag
        self.content = content
        self.children = list(children)

    def bracket(self) -> str:
        """Represents the tree in bracket notation."""
        content_str = f', "{self.content}"' if self.tag in ('td', 'th') else ''
        children_str = "".join(child.bracket() for child in self.children)
        return f"{{{self.tag}{content_str}{children_str}}}"


class _TableTreeConfig(Config):
    """A custom configuration for the APTED algorithm defining node operation costs."""

    def rename(self, node1: _TableTree, node2: _TableTree) -> float:
        """Calculates the cost of renaming one node to another."""
        if node1.tag != node2.tag:
            return 1.0
        if node1.tag in ('td', 'th'):
            return self.normalized_distance(node1.content, node2.content)
        return 0.0

    @staticmethod
    def _maximum(*sequences: str) -> int:
        """Gets the length of the longest sequence."""
        return max(map(len, sequences)) if sequences else 0

    def normalized_distance(self, seq1: str, seq2: str) -> float:
        """Calculates Levenshtein distance, normalized to a [0, 1] range."""
        max_len = self._maximum(seq1, seq2)
        if max_len == 0:
            return 0.0
        return distance.levenshtein(seq1, seq2) / max_len


class MarkdownTableSimilarity:
    """
    Calculates a comprehensive similarity score between Markdown documents based on their tables.

    This class implements a multi-stage pipeline:
    1. Preprocesses raw strings to handle special tokens from thinking models.
    2. Extracts all table-like sections from the cleaned Markdown strings.
    3. Converts each table into a robust HTML representation and then into a tree structure.
    4. Merges fragmented tables that are semantically similar using a fuzzy header matching heuristic.
    5. Computes a similarity matrix between the sets of ground truth and predicted tables using
       the Tree-Edit-Distance-based Similarity (TEDS) score for each pair.
    6. Solves the assignment problem using the Hungarian algorithm to find the optimal global matching.
    7. Returns a final score from 0.0 to 1.0, representing the overall document table similarity.
    """
    DEFAULT_MERGE_THRESHOLD = 0.8

    def __init__(self, merge_threshold: float = DEFAULT_MERGE_THRESHOLD):
        """
        Initializes the similarity calculator.

        Args:
            merge_threshold (float): The similarity score [0.0, 1.0] above which
                                     consecutive table headers are considered a match
                                     for merging. Defaults to 0.8.
        """
        self.merge_threshold = merge_threshold
        self._md_parser = MarkdownIt('commonmark').enable('table')

    def _preprocess_model_output(self, markdown_string: str) -> str:
        """
        Cleans the raw markdown output from a thinking model.
        """
        # Handle case where output was truncated mid-thought
        if '<think>' in markdown_string and '</think>' not in markdown_string:
            return ""

        # Remove analysis and think blocks
        processed_string = re.sub(r'<analysis>.*?</analysis>', '', markdown_string, flags=re.DOTALL)
        processed_string = re.sub(r'<think>.*?</think>', '', processed_string, flags=re.DOTALL)
        
        # Extract content from answer block if it exists
        answer_match = re.search(r'<answer>(.*?)</answer>', processed_string, flags=re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        return processed_string.strip()

    def _extract_raw_tables(self, markdown_string: str) -> List[str]:
        """Manually extracts table sections from a markdown string."""
        tables, current_lines = [], []
        in_table = False
        for line in markdown_string.strip().split('\n'):
            line_is_table_part = '|' in line and ('---' in line or len(line.strip()) > 2)
            if line_is_table_part:
                current_lines.append(line)
                in_table = True
            else:
                if in_table:
                    tables.append("\n".join(current_lines))
                    current_lines = []
                in_table = False
        if current_lines:
            tables.append("\n".join(current_lines))
        return tables

    def _build_tree_from_markdown(self, md_table_string: str) -> Optional[_TableTree]:
        """Renders a single markdown table string to HTML and builds a tree."""
        html_string = self._md_parser.render(md_table_string)
        lxml_tree = html.fromstring(html_string)
        try:
            lxml_table = lxml_tree.xpath('//table')[0]
            return self._convert_lxml_to_tree(lxml_table)
        except IndexError:
            return None

    def _convert_lxml_to_tree(self, lxml_element) -> _TableTree:
        """Recursively converts an lxml element to our custom _TableTree."""
        content = lxml_element.text_content().strip() if lxml_element.tag in ('th', 'td') else ""
        return _TableTree(
            lxml_element.tag,
            content,
            *[self._convert_lxml_to_tree(child) for child in lxml_element.getchildren()]
        )

    def _get_header_similarity(self, header1: _TableTree, header2: _TableTree) -> float:
        """Calculates similarity between two header rows by averaging cell-by-cell scores."""
        cells1, cells2 = header1.children, header2.children
        if len(cells1) != len(cells2) or not cells1:
            return 0.0

        config = _TableTreeConfig()
        scores = [1.0 - config.normalized_distance(c1.content, c2.content) for c1, c2 in zip(cells1, cells2)]
        return sum(scores) / len(scores)

    def _merge_tables(self, tables: List[_TableTree]) -> List[_TableTree]:
        """Merges consecutive tables in a list if their headers are sufficiently similar."""
        if len(tables) <= 1:
            return tables
        
        merged_tables, i = [], 0
        while i < len(tables):
            base_table = tables[i]
            i += 1
            if not base_table: continue
            
            try:
                tbody = next(c for c in base_table.children if c.tag == 'tbody')
                body_rows = [r for r in tbody.children if r.tag == 'tr']
                thead = next(c for c in base_table.children if c.tag == 'thead')
                base_header_row = next(c for c in thead.children if c.tag == 'tr')
            except StopIteration:
                merged_tables.append(base_table)
                continue

            while i < len(tables):
                next_table = tables[i]
                if not next_table:
                    i += 1
                    continue
                try:
                    next_tbody = next(c for c in next_table.children if c.tag == 'tbody')
                    next_thead = next(c for c in next_table.children if c.tag == 'thead')
                    next_header_row = next(c for c in next_thead.children if c.tag == 'tr')
                    if self._get_header_similarity(base_header_row, next_header_row) < self.merge_threshold:
                        break
                    body_rows.extend(r for r in next_tbody.children if r.tag == 'tr')
                    i += 1
                except StopIteration:
                    break

            new_thead = _TableTree('thead', "", base_header_row)
            new_tbody = _TableTree('tbody', "", *body_rows)
            merged_tables.append(_TableTree('table', "", new_thead, new_tbody))
            
        return merged_tables

    def _calculate_teds(self, tree1: _TableTree, tree2: _TableTree) -> float:
        """Computes the TEDS score between two table trees."""
        nodes1 = self._count_nodes(tree1)
        nodes2 = self._count_nodes(tree2)
        n_nodes = max(nodes1, nodes2)
        
        if n_nodes == 0:
            return 1.0
            
        distance = APTED(tree1, tree2, _TableTreeConfig()).compute_edit_distance()
        return 1.0 - (distance / n_nodes)

    def _count_nodes(self, node: Optional[_TableTree]) -> int:
        """Recursively counts the number of nodes in a tree."""
        if not node: return 0
        return 1 + sum(self._count_nodes(c) for c in node.children)

    def evaluate(self, pred_md: str, true_md: str) -> float:
        """
        Calculates the similarity between the tables in two markdown documents.

        Args:
            pred_md (str): The predicted markdown string.
            true_md (str): The ground truth markdown string.

        Returns:
            float: A similarity score between 0.0 and 1.0.
        """
        pred_md_processed = self._preprocess_model_output(pred_md)
        true_md_processed = self._preprocess_model_output(true_md)

        raw_pred_tables = self._extract_raw_tables(pred_md_processed)
        raw_true_tables = self._extract_raw_tables(true_md_processed)
        
        pred_trees_raw = [self._build_tree_from_markdown(t) for t in raw_pred_tables]
        true_trees_raw = [self._build_tree_from_markdown(t) for t in raw_true_tables]

        pred_trees = [tree for tree in pred_trees_raw if tree is not None]
        true_trees = [tree for tree in true_trees_raw if tree is not None]
        
        pred_tables = self._merge_tables(pred_trees)
        true_tables = self._merge_tables(true_trees)

        if not true_tables and not pred_tables: return 1.0
        if not true_tables or not pred_tables: return 0.0

        cost_matrix = np.zeros((len(true_tables), len(pred_tables)))
        for i, tree_true in enumerate(true_tables):
            for j, tree_pred in enumerate(pred_tables):
                similarity = self._calculate_teds(tree_true, tree_pred)
                cost_matrix[i, j] = 1.0 - similarity

        true_indices, pred_indices = linear_sum_assignment(cost_matrix)

        total_similarity = sum(1.0 - cost_matrix[i, j] for i, j in zip(true_indices, pred_indices))
        num_total_tables = max(len(true_tables), len(pred_tables))
        return total_similarity / num_total_tables if num_total_tables > 0 else 1.0

def run_evaluation():
    """
    Main function to execute the Markdown TEDS evaluation pipeline.
    """
    # --- Configuration ---
    expected_csv_path = "eval_selected_output.csv"
    actual_csv_path = "eval_qwen_32b.csv"
    output_results_path = "evaluation_results_teds.csv"

    # --- Load Data ---
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

    # --- Initialize Calculator ---
    teds_calculator = MarkdownTableSimilarity(merge_threshold=0.8)

    # --- Run Evaluation ---
    print(f"Starting TEDS evaluation for {len(df_expected)} items...")
    
    teds_scores = [
        teds_calculator.evaluate(pred_md=act_row, true_md=exp_row)
        for exp_row, act_row in zip(df_expected['output'], df_actual['output'])
    ]
    
    print("TEDS evaluation complete.")

    # --- Analyze and Display Results ---
    average_teds_score = (sum(teds_scores) / len(teds_scores)) * 100
    
    print("\nTEDS Evaluation Summary:")
    print("=" * 50)
    print(f"{'Average Markdown TEDS Score':<35}: {average_teds_score:.2f}%")
    print("=" * 50)

    # --- Save Detailed Results ---
    results_df = pd.DataFrame({'teds_score': teds_scores})
    results_df.index.name = "row_index"
    try:
        results_df.to_csv(output_results_path)
        print(f"\nDetailed TEDS scores saved to '{output_results_path}'")
    except Exception as e:
        print(f"\nError saving detailed results: {e}")

if __name__ == "__main__":
    run_evaluation()