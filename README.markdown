# DSPy-ASP Spatial Reasoning Pipeline

A DSPy-based pipeline for spatial reasoning, converting natural language descriptions and questions into Answer Set Programming (ASP) facts/queries, solved with Clingo. Includes iterative refinement and parallel evaluation.

## Features
- Converts NLP to ASP for spatial QA tasks.
- Supports multiple LLMs (OpenAI, DeepSeek, Llama).
- Iterative ASP error refinement.
- Parallel evaluation with accuracy metrics.

## Installation
1. Clone repo:
   ```bash
   git clone https://github.com/yourusername/dspy-asp-spatial-reasoning.git
   cd dspy-asp-spatial-reasoning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add API keys in `config.py` (e.g., OpenRouter key).
4. Install Clingo:
   ```bash
   # macOS
   brew install clingo
   # Ubuntu
   sudo apt install clingo
   ```

## Usage
Evaluate on a dataset:
```bash
python elevate.py
```
- Uses `clean/qa10_test.json` (limit: 300 examples).
- Outputs to `results/deepseek_q10.json`.

Run pipeline directly:
```python
from pipeline import Pipeline
pipeline = Pipeline(max_iterations=3)
result = pipeline(context="Block a has a red circle above a blue square.", question="Is the red circle above the blue square?")
print(result.answer)  # e.g., "yes"
```

## Project Structure
- `config.py`: Model configs, API keys, ASP rules.
- `pipeline.py`: NLP-to-ASP conversion and refinement.
- `solver.py`: Clingo ASP solver wrapper.
- `elevate.py`: Evaluation and optimization script.

## Requirements
- Python 3.10+
- See `requirements.txt` for dependencies.

## License
MIT License (see `LICENSE`).