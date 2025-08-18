import clingo
import sys
from io import StringIO
import re
from functools import lru_cache

@lru_cache(maxsize=1000)
def run_asp_and_get_answer(asp_program: str) -> dict:
    """
    Runs an ASP program using Clingo and extracts results from `query` predicates.
    Args:
        asp_program (str): A string containing a valid ASP program with `query(X)` predicates.
    Returns:
        dict: A dictionary with:
            - success (bool): True if the program is satisfiable and has results.
            - answer (str or None): Comma-separated `query` results or single result.
            - error (str or None): Error message if the program fails or has no results.
    """
    results = []
    error = None
    answer = None
    success = False
    asp_program = asp_program.replace("```asp", "").replace("```", "").replace("`", "").replace("prolog", "").replace("asp", "").strip()

    # Optional: Remove extra blank lines
    asp_program = re.sub(r'\n\s*\n', '\n', asp_program)


    stderr_capture = StringIO()
    original_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
        ctl = clingo.Control()
        ctl.add("base", [], asp_program)
        ctl.ground([("base", [])])

        def on_model(model):
            for atom in model.symbols(shown=True):
                if atom.name == "query" and atom.arguments:
                    results.append(str(atom.arguments[0]))

        solve_result = ctl.solve(on_model=on_model)

        if solve_result.satisfiable:
            if results:
                answer = ", ".join(results) if len(results) > 1 else results[0]
                success = True
            else:
                error = "Satisfiable, but no query results"
        else:
            error = "Program is unsatisfiable"
    except Exception as e:
        error = f"Clingo error: {str(e)}"
    finally:
        sys.stderr = original_stderr
        stderr_output = stderr_capture.getvalue().strip()
        stderr_capture.close()
        if stderr_output and not error:
            error = f"Clingo stderr: {stderr_output}"
        elif stderr_output and error:
            error += f"\nAdditional stderr: {stderr_output}"

    return {
        "success": success,
        "answer": answer,
        "error": error
    }