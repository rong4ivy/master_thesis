
"""
Final version of compare_optimizers with proper DSPy optimizer configurations
based on official DSPy documentation: https://dspy.ai/cheatsheet/
"""
import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShot, COPRO, LabeledFewShot
from dspy.evaluate import Evaluate
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
import multiprocessing as mp
from collections import Counter
from pipeline import Pipeline
from fuzzywuzzy import fuzz


def answer_accuracy_metric(example, pred, trace=None):
    """
    DSPy-compatible accuracy metric.
    Args:
        example: DSPy Example object with .answer attribute
        pred: DSPy Prediction object with .answer attribute  
        trace: Optional trace (not used)
    Returns:
        float: 0.0, 0.5, or 1.0
    """
    # Extract answers from DSPy objects
    true_answer = getattr(example, 'answer', '') or str(example.get('answer', ''))
    pred_answer = getattr(pred, 'answer', '') or str(pred)
    
    # Clean and normalize
    pred_answer = str(pred_answer).strip().lower()
    true_answer = str(true_answer).strip().lower()
    
    # Split into items
    golds = sorted([x.strip() for x in true_answer.split(',') if x.strip()])
    preds = sorted([x.strip() for x in pred_answer.split(',') if x.strip()])
    
    if not golds or not preds:
        return 0.0
    
    # Fuzzy matching with threshold
    threshold = 60
    gold_matches = [any(fuzz.ratio(g, p) > threshold for p in preds) for g in golds]
    num_matched = sum(gold_matches)
    
    # Scoring
    if num_matched == len(golds) and len(preds) == len(golds):
        return 1.0  # Perfect match
    elif num_matched > 0:
        return 0.5  # Partial match
    else:
        return 0.0  # No match


def load_examples(file_path: str, question_type: str = "YN", limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load examples from CSV file."""
    df = pd.read_csv(file_path)
    df = df[df["Q_type"] == question_type]
    
    if limit:
        df = df[:limit]
    
    return [
        {
            "context": row["Story"],
            "question": row["Question"],
            "answer": row["Answer"].strip().lower()
        } for _, row in df.iterrows()
    ]


def convert_to_dspy_examples(examples: List[Dict], input_keys: List[str] = None):
    """Convert dict examples to DSPy Example objects."""
    if input_keys is None:
        input_keys = ["context", "question"]
    
    dspy_examples = []
    for ex in examples:
        example = dspy.Example(**ex).with_inputs(*input_keys)
        dspy_examples.append(example)
    
    return dspy_examples


def optimize_with_mipro(pipeline, trainset, valset=None):
    """
    MIPROv2 optimizer - based on DSPy documentation.
    Automatically optimizes both instructions and few-shot examples.
    """
    print("  🔧 Optimizing with MIPROv2...")
    
    # Convert to DSPy format
    train_examples = convert_to_dspy_examples(trainset)
    val_examples = convert_to_dspy_examples(valset) if valset else None
    
    try:
        # MIPROv2 configuration based on DSPy docs
        optimizer = MIPROv2(
            metric=answer_accuracy_metric,
            auto="light",  # "light", "medium", or "heavy"
            num_candidates=10,  # Number of candidates to generate
            init_temperature=1.0,
            verbose=True
        )
        
        # Compile with train and optional validation set
        compile_kwargs = {
            "student": pipeline,
            "trainset": train_examples,
            "requires_permission_to_run": False
        }
        
        if val_examples:
            compile_kwargs["valset"] = val_examples
        
        optimized = optimizer.compile(**compile_kwargs)
        print("  MIPROv2 optimization completed")
        return optimized
        
    except Exception as e:
        print(f"  MIPROv2 optimization failed: {e}")
        return pipeline


def optimize_with_bootstrap(pipeline, trainset):
    """
    BootstrapFewShot optimizer - based on DSPy documentation.
    Generates few-shot examples automatically.
    """
    print("  🔧 Optimizing with BootstrapFewShot...")
    
    # Convert to DSPy format
    train_examples = convert_to_dspy_examples(trainset)
    
    try:
        # BootstrapFewShot configuration based on DSPy docs
        optimizer = BootstrapFewShot(
            metric=answer_accuracy_metric,
            max_bootstrapped_demos=4,    # Max few-shot examples to generate
            max_labeled_demos=4,         # Max labeled examples to use
            max_rounds=2,                # Max optimization rounds
            max_errors=3,                # Max errors before stopping
            verbose=True
        )
        
        optimized = optimizer.compile(
            student=pipeline,
            trainset=train_examples
        )
        
        print("  BootstrapFewShot optimization completed")
        return optimized
        
    except Exception as e:
        print(f"   BootstrapFewShot optimization failed: {e}")
        return pipeline


def optimize_with_copro(pipeline, trainset, valset=None):
    """
    COPRO optimizer - based on DSPy documentation.
    Optimizes prompts using coordinate ascent.
    """
    print("  🔧 Optimizing with COPRO...")
    
    # Convert to DSPy format
    train_examples = convert_to_dspy_examples(trainset)
    val_examples = convert_to_dspy_examples(valset) if valset else None
    
    try:
        # COPRO configuration based on DSPy docs
        optimizer = COPRO(
            metric=answer_accuracy_metric,
            breadth=10,                  # Number of candidates per step
            depth=3,                     # Number of optimization steps
            init_temperature=1.4,        # Initial temperature for sampling
            verbose=True
        )
        
        # COPRO compile arguments
        compile_kwargs = {
            "student": pipeline,
            "trainset": train_examples
        }
        
        # Add validation set if available
        if val_examples:
            compile_kwargs["eval_kwargs"] = {"devset": val_examples}
        
        optimized = optimizer.compile(**compile_kwargs)
        print("  COPRO optimization completed")
        return optimized
        
    except Exception as e:
        print(f"  COPRO optimization failed: {e}")
        return pipeline


def optimize_with_labeled_fewshot(pipeline, trainset):
    """
    LabeledFewShot optimizer - based on DSPy documentation.
    Simple few-shot learning with labeled examples.
    """
    print("  🔧 Optimizing with LabeledFewShot...")
    
    # Convert to DSPy format
    train_examples = convert_to_dspy_examples(trainset)
    
    try:
        # LabeledFewShot configuration based on DSPy docs
        optimizer = LabeledFewShot(k=4)  # Use 4 few-shot examples
        
        optimized = optimizer.compile(
            student=pipeline,
            trainset=train_examples
        )
        
        print("   LabeledFewShot optimization completed")
        return optimized
        
    except Exception as e:
        print(f"  LabeledFewShot optimization failed: {e}")
        return pipeline


def get_optimizer_function(optimizer_type: str):
    """Get the appropriate optimizer function."""
    optimizers = {
        'mipro': optimize_with_mipro,
        'bootstrap': optimize_with_bootstrap,
        'copro': optimize_with_copro,
        'labeled_fewshot': optimize_with_labeled_fewshot
    }
    return optimizers.get(optimizer_type)


def evaluate_pipeline_performance(pipeline, testset, verbose=False):
    """Evaluate pipeline performance on test set."""
    results = []
    
    for i, example in enumerate(testset):
        if verbose and i % 5 == 0:
            print(f"    Evaluating example {i+1}/{len(testset)}...")
        
        try:
            prediction = pipeline(context=example["context"], question=example["question"])
            predicted = str(getattr(prediction, 'answer', "")).strip().lower()
            
            # Calculate accuracy using our metric
            dspy_example = dspy.Example(answer=example["answer"])
            dspy_pred = dspy.Prediction(answer=predicted)
            accuracy = answer_accuracy_metric(dspy_example, dspy_pred)
            
            result = {
                "question": example["question"],
                "expected": example["answer"],
                "predicted": predicted,
                "success": getattr(prediction, 'success', False),
                "accurate": accuracy,
                "iterations": getattr(prediction, 'iterations', 0),
                "final_asp_code": getattr(prediction, 'final_asp_code', '')
            }
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "question": example["question"],
                "expected": example["answer"],
                "predicted": "",
                "success": False,
                "accurate": 0.0,
                "iterations": 0,
                "final_asp_code": "",
                "error": str(e)
            })
    
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate comprehensive metrics from results."""
    if not results:
        return {"total": 0, "success_rate": 0.0, "accuracy_rate": 0.0, "both_rate": 0.0}
    
    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    accuracy_count = sum(1 for r in results if r["accurate"] > 0)
    both_count = sum(1 for r in results if r["success"] and r["accurate"] > 0)
    
    return {
        "total": total,
        "success_count": success_count,
        "accuracy_count": accuracy_count,
        "both_count": both_count,
        "success_rate": success_count / total,
        "accuracy_rate": accuracy_count / total,
        "both_rate": both_count / total,
        "mean_accuracy": sum(r["accurate"] for r in results) / total
    }


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


def compare_optimizers(
    input_file: str,
    output_dir: str = "results/optimizer_comparison",
    optimizers: List[str] = None,
    question_type: str = "YN",
    limit: int = 300,
    train_ratio: float = 0.3,
    max_train_size: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Compare different DSPy optimizers on the pipeline.
    
    Args:
        input_file: Path to CSV data file
        output_dir: Directory to save results
        optimizers: List of optimizer names to compare
        question_type: Type of questions to filter ("YN", "FR", etc.)
        limit: Maximum number of examples to use
        train_ratio: Ratio of data to use for training
        max_train_size: Maximum training examples (to prevent timeout)
        verbose: Whether to show detailed progress
        
    Returns:
        Dict containing all results and comparisons
    """
    
    # Default optimizers to compare
    if optimizers is None:
        optimizers = ['baseline', 'mipro', 'bootstrap', 'copro']
    
    print(f"🚀 DSPy Optimizer Comparison")
    print(f"{'='*60}")
    print(f"📁 Input file: {input_file}")
    print(f"📊 Question type: {question_type}")
    print(f"🔢 Limit: {limit} examples")
    print(f"🎯 Optimizers: {optimizers}")
    
    # Load and split data
    examples = load_examples(input_file, question_type=question_type, limit=limit)
    if not examples:
        print("No examples loaded")
        return {}
    
    # Split data
    train_size = min(max_train_size, max(5, int(len(examples) * train_ratio)))
    trainset = examples[:train_size]
    testset = examples[train_size:]
    
    # Create small validation set from training data
    val_size = min(5, len(trainset) // 2)
    valset = trainset[-val_size:] if val_size > 0 else None
    trainset = trainset[:-val_size] if valset else trainset
    
    print(f"📊 Data split:")
    print(f"   Total: {len(examples)} examples")
    print(f"   Training: {len(trainset)} examples")
    print(f"   Validation: {len(valset) if valset else 0} examples")
    print(f"   Testing: {len(testset)} examples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare each optimizer
    all_results = {}
    comparison_summary = {}
    
    for optimizer_name in optimizers:
        print(f"\n{'='*60}")
        print(f"🔧 Testing {optimizer_name.upper()} Optimizer")
        print(f"{'='*60}")
        
        try:
            # Create fresh pipeline
            pipeline = Pipeline(max_iterations=3)
            
            # Optimize pipeline (skip for baseline)
            if optimizer_name != 'baseline':
                optimizer_func = get_optimizer_function(optimizer_name)
                if not optimizer_func:
                    print(f" Unknown optimizer: {optimizer_name}")
                    continue
                
                # Apply optimization
                if optimizer_name in ['mipro', 'copro']:
                    pipeline = optimizer_func(pipeline, trainset, valset)
                else:
                    pipeline = optimizer_func(pipeline, trainset)
            else:
                print("  📝 Using baseline pipeline (no optimization)")
            
            # Evaluate pipeline
            print(f"  🧪 Evaluating on {len(testset)} test examples...")
            results = evaluate_pipeline_performance(pipeline, testset, verbose=verbose)
            
            # Calculate metrics
            metrics = calculate_metrics(results)
            cumulative_metrics = calculate_cumulative_metrics(results)
            
            # Store results
            optimizer_data = {
                "optimizer": optimizer_name,
                "results": results,
                "metrics": metrics,
                "cumulative_metrics": cumulative_metrics,
                "config": {
                    "train_size": len(trainset),
                    "val_size": len(valset) if valset else 0,
                    "test_size": len(testset)
                }
            }
            
            all_results[optimizer_name] = optimizer_data
            comparison_summary[optimizer_name] = {
                "success_rate": metrics["success_rate"],
                "accuracy_rate": metrics["accuracy_rate"], 
                "both_rate": metrics["both_rate"],
                "mean_accuracy": metrics["mean_accuracy"]
            }
            
            # Print results
            print(f"  ✅ {optimizer_name.upper()} Results:")
            print(f"     Success Rate: {metrics['success_rate']:.1%} ({metrics['success_count']}/{metrics['total']})")
            print(f"     Accuracy Rate: {metrics['accuracy_rate']:.1%} ({metrics['accuracy_count']}/{metrics['total']})")
            print(f"     Both Success & Accurate: {metrics['both_rate']:.1%} ({metrics['both_count']}/{metrics['total']})")
            print(f"     Mean Accuracy Score: {metrics['mean_accuracy']:.3f}")
            
            # Save individual results
            output_file = os.path.join(output_dir, f"{optimizer_name}_detailed_results.json")
            with open(output_file, 'w') as f:
                json.dump(optimizer_data, f, indent=2)
            print(f"     💾 Detailed results saved to: {output_file}")
            
        except Exception as e:
            print(f"  ❌ {optimizer_name.upper()} failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Print final comparison
    print(f"\n{'='*80}")
    print("🏆 FINAL OPTIMIZER COMPARISON")
    print(f"{'='*80}")
    
    if comparison_summary:
        # Sort by both_rate (best overall metric)
        sorted_optimizers = sorted(
            comparison_summary.items(),
            key=lambda x: x[1]["both_rate"],
            reverse=True
        )
        
        print(f"{'Optimizer':<15} | {'Success':>8} | {'Accuracy':>8} | {'Both':>8} | {'Avg Score':>9}")
        print(f"{'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}")
        
        for i, (name, metrics) in enumerate(sorted_optimizers):
            badge = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{name:<15} | {metrics['success_rate']:7.1%} | {metrics['accuracy_rate']:7.1%} | "
                  f"{metrics['both_rate']:7.1%} | {metrics['mean_accuracy']:8.3f} {badge}")
        
        # Show best optimizer details
        best_optimizer, best_metrics = sorted_optimizers[0]
        print(f"\n🏆 Best Overall: {best_optimizer.upper()}")
        print(f"   🎯 Both Success & Accurate: {best_metrics['both_rate']:.1%}")
        
        # Show cumulative metrics for best optimizer
        if best_optimizer in all_results:
            print(f"\n📈 Cumulative Performance - {best_optimizer.upper()}:")
            cumulative = all_results[best_optimizer]["cumulative_metrics"]
            for iteration, cum_metrics in cumulative.items():
                print(f"   Iteration {iteration}: Success {cum_metrics['success']} | "
                      f"Accuracy {cum_metrics['accuracy']} | Both {cum_metrics['both']}")
    
    # Save comparison summary
    summary_file = os.path.join(output_dir, "optimizer_comparison_summary.json")
    summary_data = {
        "comparison_summary": comparison_summary,
        "best_optimizer": max(comparison_summary.keys(), key=lambda k: comparison_summary[k]["both_rate"]) if comparison_summary else None,
        "experiment_config": {
            "input_file": input_file,
            "question_type": question_type,
            "total_examples": len(examples),
            "train_size": len(trainset),
            "val_size": len(valset) if valset else 0,
            "test_size": len(testset),
            "optimizers_tested": list(comparison_summary.keys())
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    print(f"📁 All results saved in: {output_dir}")
    
    return all_results


if __name__ == "__main__":
    # Run comprehensive optimizer comparison
    results = compare_optimizers(
        input_file="spar_300_test.csv",
        output_dir="results/dspy_optimizer_comparison",
        optimizers=['baseline', 'mipro', 'bootstrap', 'copro', 'labeled_fewshot'],
        question_type="YN",
        limit=300,
        train_ratio=0.3,
        max_train_size=15,
        verbose=True
    )
    
    print("\n🎉 Optimizer comparison completed!")