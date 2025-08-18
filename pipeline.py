
"""
Main ASP-NLP Pipeline using DSPy.
"""
from config import standard_rules, convert_instructions, refine_instructions
import dspy
from solver import run_asp_and_get_answer

class Pipeline(dspy.Module):
    """Main ASP-NLP Pipeline."""
    
    def __init__(self, max_iterations=3):
        super().__init__()
        self.max_iterations = max_iterations
        
        # Create DSPy modules
        convert_sig = dspy.Signature("context, question -> facts_query", instructions=convert_instructions)
        refine_sig = dspy.Signature("context, question, asp_code, error -> refined_asp_code", instructions=refine_instructions)
        
        self.convert = dspy.ChainOfThought(convert_sig)
        self.refine = dspy.ChainOfThought(refine_sig)

    def forward(self, context, question):
        """Process context and question through the pipeline."""
        # Step 1: Convert context + question to initial facts_query
        try:
            facts_query = self.convert(context=context, question=question).facts_query
        except Exception as e:
            return dspy.Prediction(
                initial_asp_code="", final_asp_code="", answer=None,
                success=False, iterations=0, error=f"Convert error: {e}"
            )

        # Store initial ASP code for debugging
        initial_asp_code = f"{facts_query}\n{standard_rules}"
        asp_code = initial_asp_code
        result = {}

        # Step 2: Iterative solving and refining
        for i in range(self.max_iterations):
            result = run_asp_and_get_answer(asp_code)

            if result.get("success"):
                break

            if i < self.max_iterations - 1:
                try:
                    asp_code = self.refine(
                        context=context, question=question,
                        asp_code=asp_code, error=result.get("error", "Unknown error")
                    ).refined_asp_code
                except Exception as e:
                    result["error"] = f"Refine error: {e}"
                    break

        # Step 3: Return final prediction
        return dspy.Prediction(
            initial_asp_code=initial_asp_code, final_asp_code=asp_code,
            answer=result.get("answer"), success=result.get("success", False),
            iterations=i + 1, error=result.get("error")
        )

