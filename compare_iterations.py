
from dspy.teleprompt import MIPROv2, BootstrapFewShot, COPRO
from dspy.evaluate import Evaluate
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
import multiprocessing as mp
from collections import Counter
from pipeline import Pipeline
from fuzzywuzzy import fuzz

def answer_accuracy_metric(true_answer, pred_answer, trace=None):
    """Calculate accuracy metric for DSPy optimization."""
    pred_answer = str(pred_answer).strip().lower()
    true_answer = str(true_answer).strip().lower()

    golds = sorted([x.strip() for x in true_answer.split(',') if x.strip()])
    preds= sorted([x.strip() for x in pred_answer.split(',') if x.strip()])
    
    threshold = 60
  
    if not golds or not preds:
        return 0.0

    # Check if every gold has a fuzzy match in preds
    gold_matches = [any(fuzz.ratio(g, p) > threshold for p in preds) for g in golds]
    num_matched = sum(gold_matches)

    # Case 1: all golds matched and pred doesn't add extra noise
    if num_matched == len(golds) and len(preds) == len(golds):
        return 1.0

    # Case 2: partial match (at least one match)
    if num_matched > 0:
        return 0.5

    # Case 3: no match
    return 0.0


    # if golds == preds:
    #     return 1.0
    # elif any(x in pred_items for x in exp_items):
    #     return 0.5
    # else:
    #     return 0.0

def load_examples(file_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load examples from CSV file."""
    df = pd.read_csv(file_path)
    df = df[df["Q_type"] == "FR"]  # Filter early
    
    if limit:
        df = df.head(limit)
    return [
        {
            "context": row["Story"],
            "question": row["Question"], 
            "answer": row["Answer"].strip().lower()
        } for _, row in df.iterrows()
    ]
    

def process_batch(batch: List[Dict]) -> List[Dict]:
    """Process a batch of examples."""

    pipeline = Pipeline(max_iterations=3)
    results = []
    
    for example in batch:
        try:
            prediction = pipeline(context=example["context"], question=example["question"])
            predicted = str(getattr(prediction, 'answer', "")).strip().lower()
            results.append({
                "question": example["question"],
                "expected": example["answer"],
                "final_asp_code": getattr(prediction, 'final_asp_code', ''),
                "predicted": predicted,
                "success": getattr(prediction, 'success', False),
                "accurate": answer_accuracy_metric(example["answer"], predicted),
               
                "iterations": getattr(prediction, 'iterations', 0)
            })
        except Exception as e:
            results.append({
                "question": example["question"],
                "expected": example["answer"], 
                 "final_asp_code": "",
                "predicted": "",
                "success": False,
                "accurate": 0.0,
                "iterations": 0,
                "error": str(e)
            })
    return results

def calculate_cumulative_metrics(results: List[Dict]) -> Dict[int, Dict]:
    """Calculate cumulative performance by iteration."""
    if not results:
        return {}
    
    max_iter = max(r.get("iterations", 0) for r in results)
    metrics = {}
    total = len(results)
    
    for i in range(max_iter + 1):
        success_count = sum(1 for r in results if r.get("iterations", 0) <= i and r["success"])
        accuracy_count = sum(1 for r in results if r.get("iterations", 0) <= i and r["accurate"] > 0)
        both_count = sum(1 for r in results if r.get("iterations", 0) <= i and r["success"] and r["accurate"] > 0)
        
        metrics[i] = {
            "success": f"{success_count}/{total} ({success_count/total:.1%})",
            "accuracy": f"{accuracy_count}/{total} ({accuracy_count/total:.1%})",
            "both": f"{both_count}/{total} ({both_count/total:.1%})"
        }
    return metrics

def evaluate(input_file: str, output_file: Optional[str] = None, limit: int = 50, 
             batch_size: int = 10, processes: Optional[int] = None) -> Dict:
    """Main evaluation function."""
    # Load data
    examples = load_examples(input_file, limit)
    if not examples:
        return {"error": "No examples loaded"}
    
    print(f"Processing {len(examples)} examples...")
    
    # Process in parallel
    processes = processes or max(1, mp.cpu_count() - 1)
    batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
    
    with mp.Pool(processes) as pool:
        batch_results = pool.map(process_batch, batches)
    
    # Flatten results
    all_results = [r for batch in batch_results for r in batch]
    
    # Calculate metrics
    total = len(all_results)
    success_rate = sum(r["success"] for r in all_results) / total
    accuracy_rate = sum(r["accurate"] for r in all_results) / total
    both_rate = sum(r["success"] and r["accurate"] > 0 for r in all_results) / total
    
    # Print results
    print(f"\n=== RESULTS ===")
    print(f"Overall Success: {success_rate:.1%}")
    print(f"Overall Accuracy: {accuracy_rate:.1%}")
    print(f"Both: {both_rate:.1%}")
    
    print(f"\n=== CUMULATIVE BY ITERATION ===")
    for iteration, metrics in calculate_cumulative_metrics(all_results).items():
        print(f"By iteration {iteration}:")
        print(f"  Success: {metrics['success']}")
        print(f"  Accuracy: {metrics['accuracy']}")
        print(f"  Both: {metrics['both']}")
    
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {output_file}")
    
    return {"success_rate": success_rate, "accuracy_rate": accuracy_rate, "both_rate": both_rate, "results": all_results}

if __name__ == "__main__":
    evaluate(
        input_file="spar_300_test.csv",
        output_file="results/deepseek_FR.json",
        limit=300,
        batch_size=10,
        processes=30
    )
