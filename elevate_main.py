
"""
The main code to evaluate the performance of DSPy based pipeline
"""

from pipeline import Pipeline
import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
from typing import List, Dict, Any
import re
from functools import lru_cache
import hashlib
from fuzzywuzzy import fuzz
import os
import random
from collections import Counter 




def load_examples(json_path, limit=None):
    """Fast example loading"""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    items = list(data.values()) if isinstance(data, dict) else data
    if limit:
        items = items[:limit]
    
    examples = []
    for item in items:
        try:
            context = " ".join(item["story"]) if isinstance(item["story"], list) else item["story"]
            examples.append(dspy.Example(
                context=context,
                question=item["question"], 
                answer=item["label"]
            ).with_inputs("context", "question"))
        except KeyError:
            continue
    return examples

def process_single(example, pipeline):
    """Process one example with detailed results"""
    try:
        prediction = pipeline(context=example.context, question=example.question)
        
        pred_answer = str(getattr(prediction, 'answer', "")).strip().lower()
        true_answer = str(example.answer).strip().lower()
        accurate = fuzz.ratio(pred_answer, true_answer) > 80
        
        return {
            "question": example.question,
            "context": example.context,
            "expected": true_answer,
            "predicted": pred_answer,
            "success": getattr(prediction, 'success', False),
            "accurate": accurate,
            "iterations": getattr(prediction, 'iterations', 0),
            "error": getattr(prediction, 'error', None),
            "initial_asp_code": getattr(prediction, 'initial_asp_code', '') ,
            "final_asp_code": getattr(prediction, 'final_asp_code', '') ,
            }
    except Exception as e:
        return {
            "question": getattr(example, 'question', 'N/A'),
            "context": "Error",
            "expected": str(getattr(example, 'answer', '')),
            "predicted": "",
            "success": False,
            "accurate": False, 
            "iterations": 0,
            "error": str(e),
            "initial_asp_code": "",
            "final_asp_code": ""
        }


def answer_accuracy_metric(true_answer, pred_answer, trace=None):
    pred_answer = str(pred_answer).strip().lower()
    true_answer = str(true_answer).strip().lower()

    exp_items = sorted([x.strip() for x in true_answer.split(',') if x.strip()])
    pred_items = sorted([x.strip() for x in pred_answer.split(',') if x.strip()])

    if exp_items == pred_items:
        return 1.0
    elif any(x in pred_items for x in exp_items):
        return 0.5
    else:
        return 0.0
    
    
def process_batch(job: Dict[str, Any]) -> List[Dict[str, Any]]:
 
    examples: List[dspy.Example] = job.get('examples', []) or []


    pipeline = Pipeline(max_iterations=3)

    optimizer = MIPROv2(metric=answer_accuracy_metric, auto='light', verbose=False)
    trainset = examples[:5]
    # compile returns a compatible pipeline instance
    opt_pipeline= optimizer.compile(pipeline, trainset=trainset, requires_permission_to_run=False)

    results: List[Dict[str, Any]] = []
    for ex in examples:
        results.append(process_single(ex,  opt_pipeline))

    return results


def calculate_cumulative_metrics(results: List[Dict]) -> Dict[int, Dict]:
    """Calculate cumulative performance by iteration"""
    if not results:
        return {}
    
    max_iter = max(r.get("iterations", 0) for r in results)
    metrics = {}
    total = len(results)
    
    for i in range(max_iter + 1):
        # Count results up to and including iteration i
        success_count = sum(1 for r in results if r.get("iterations", 0) <= i and r["success"])
        accuracy_count = sum(1 for r in results if r.get("iterations", 0) <= i and r["accurate"])
        both_count = sum(1 for r in results if r.get("iterations", 0) <= i and r["success"] and r["accurate"])
        
        metrics[i] = {
            "success": f"{success_count}/{total} ({success_count/total*100:.1f}%)",
            "accuracy": f"{accuracy_count}/{total} ({accuracy_count/total*100:.1f}%)",
            "both": f"{both_count}/{total} ({both_count/total*100:.1f}%)"
        }
    
    return metrics

def evaluate_fast(examples, pipeline, batch_size=10):
    """Fast sequential evaluation with progress tracking (original function preserved)"""
    print(f"=== EVALUATING {len(examples)} EXAMPLES (Sequential) ===")
    
    results = []
    for i, example in enumerate(examples):
        if i % batch_size == 0:
            print(f"Progress: {i}/{len(examples)} ({i/len(examples)*100:.1f}%)")
        
        result = process_single(example, pipeline)
        results.append(result)
    
    # Calculate metrics
    total = len(results)
    successful = sum(r["success"] for r in results)
    accurate = sum(r["accurate"] for r in results)
    both = sum(r["success"] and r["accurate"] for r in results)
    
    print(f"\n=== RESULTS ===")
    print(f"Total: {total}")
    print(f"ASP Success: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"Answer Accuracy: {accurate}/{total} ({accurate/total*100:.1f}%)")  
    print(f"Both Success + Accurate: {both}/{total} ({both/total*100:.1f}%)")
    
    # Show iteration distribution
    if successful > 0:
        iterations = [r["iterations"] for r in results if r["success"]]
        print(f"Success by iteration: {dict(Counter(iterations))}")
    if accurate > 0:
        iterations = [r["iterations"] for r in results if r["accurate"]]
        print(f"Accurate by iteration: {dict(Counter(iterations))}")
    
    # Show cumulative metrics
    print(f"\n=== CUMULATIVE PERFORMANCE BY ITERATION ===")
    cumulative_metrics = calculate_cumulative_metrics(results)
    for iteration, metrics in cumulative_metrics.items():
        print(f"By iteration {iteration}:")
        print(f"  Success: {metrics['success']}")
        print(f"  Accuracy: {metrics['accuracy']}")
        print(f"  Both: {metrics['both']}")
    
    # Show sample failures
    failures = [r for r in results if not (r["success"] and r["accurate"])]
    if failures:
        print(f"\nSample failures:")
        for i, f in enumerate(failures[:3]):
            print(f"  {i+1}. Expected: '{f['expected']}' Got: '{f['predicted']}'")
            if f["error"]:
                print(f"     Error: {f['error']}")
    
    return {
        "total": total,
        "success_rate": successful/total,
        "accuracy_rate": accurate/total, 
        "both_rate": both/total,
        "results": results
    }

def evaluate_parallel(examples, batch_size=10, processes=None):
    """Parallel evaluation using multiprocessing"""
    print(f"=== EVALUATING {len(examples)} EXAMPLES (Parallel) ===")
    
    # Determine number of processes
    if processes is None:
        processes = max(1, mp.cpu_count() - 1)
    print(f"Using {processes} processes with batch size {batch_size}")
    
    # Create batches
    batches = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        batches.append({'examples': batch})
    
    print(f"Created {len(batches)} batches")
    
    # Process batches in parallel
    with mp.Pool(processes) as pool:
        batch_results = pool.map(process_batch, batches)
    
    # Flatten results
    results = [r for batch in batch_results for r in batch]
    
    # Calculate metrics
    total = len(results)
    successful = sum(r["success"] for r in results)
    accurate = sum(r["accurate"] for r in results)
    both = sum(r["success"] and r["accurate"] for r in results)
    
    print(f"\n=== RESULTS ===")
    print(f"Total: {total}")
    print(f"ASP Success: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"Answer Accuracy: {accurate}/{total} ({accurate/total*100:.1f}%)")
    print(f"Both Success + Accurate: {both}/{total} ({both/total*100:.1f}%)")
    
    # Show iteration distribution
    if successful > 0:
        iterations = [r["iterations"] for r in results if r["success"]]
        print(f"Success by iteration: {dict(Counter(iterations))}")
    if accurate > 0:
        iterations = [r["iterations"] for r in results if r["accurate"]]
        print(f"Accurate by iteration: {dict(Counter(iterations))}")
    
    # Show cumulative metrics
    print(f"\n=== CUMULATIVE PERFORMANCE BY ITERATION ===")
    cumulative_metrics = calculate_cumulative_metrics(results)
    for iteration, metrics in cumulative_metrics.items():
        print(f"By iteration {iteration}:")
        print(f"  Success: {metrics['success']}")
        print(f"  Accuracy: {metrics['accuracy']}")
        print(f"  Both: {metrics['both']}")
    
    # Show sample failures
    failures = [r for r in results if not (r["success"] and r["accurate"])]
    if failures:
        print(f"\nSample failures:")
        for i, f in enumerate(failures[:3]):
            print(f"  {i+1}. Expected: '{f['expected']}' Got: '{f['predicted']}'")
            if f["error"]:
                print(f"     Error: {f['error']}")
    
    return {
        "total": total,
        "success_rate": successful/total,
        "accuracy_rate": accurate/total,
        "both_rate": both/total,
        "results": results
    }

def save_results(results, output_path):
    """Save results to JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {output_path}")
    except Exception as e:
        print(f"Failed to save: {e}")

def main(input_file: str, output_file: str = None, limit=50, use_parallel=True, 
         batch_size=10, processes=None):
    """Main function with parallel processing option"""
    print(f"Loading examples from {input_file} (limit: {limit})")
    examples = load_examples(input_file, limit=limit)
    
    if not examples:
        print("❌ No examples loaded")
        return None
    
    print(f"✓ Loaded {len(examples)} examples")
    
    # Show sample
    print(f"\nSample example:")
    print(f"Context: {examples[0].context[:100]}...")
    print(f"Question: {examples[0].question}")
    print(f"Expected: {examples[0].answer}")
    
    # Evaluate with chosen method
    if use_parallel:
        evaluation = evaluate_parallel(examples, batch_size=batch_size, processes=processes)
    else:
        pipeline = Pipeline(max_iterations=3)
        evaluation = evaluate_fast(examples, pipeline, batch_size=batch_size)
    
    # Save detailed results if requested
    if output_file:
        save_results(evaluation["results"], output_file)
    
    return evaluation

if __name__ == "__main__":
    # Quick test with parallel processing
    result = main(
        input_file="clean/qa10_test.json", 
        output_file="results/deepseek_q10.json", 
        limit=300,
        use_parallel=True,  # Enable parallel processing
        batch_size=12,      # Process 10 examples per batch
        processes=32       # Use 4 processes (or None for auto)
    )
    
    if result:
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"Success Rate: {result['success_rate']:.1%}")
        print(f"Accuracy Rate: {result['accuracy_rate']:.1%}")
        print(f"Combined Success+Accuracy: {result['both_rate']:.1%}")