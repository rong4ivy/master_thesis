
import sys
from io import StringIO
import re
from functools import lru_cache
import dspy
import logging
import threading

generation_args = {
    "temperature": 0,
    "max_tokens": 5000,
    "cache": True,        
    "track_usage": True  
    }
    
api_keys = {"openai_key": ''",
            "deepseek_key":"",
            "openrouter":"",
            "grok_key": "",
}

models = {
        'gpt4': dspy.LM("openai/chatpgpt-4o", api_key=api_keys["openai_key"], **generation_args),
      #   'deepseek': dspy.LM('openai/deepseek-chat', api_base='https://api.deepseek.com', 
      #                      api_key=api_keys["deepseek_key"], **generation_args),
         "deepseek": dspy.LM("openai/deepseek/deepseek-chat-v3-0324:free",api_base='https://openrouter.ai/api/v1', api_key=api_keys["openrouter"], **generation_args),
         
        'llama': dspy.LM('openai/meta-llama/llama-3.3-70b-instruct',
                        api_base='https://openrouter.ai/api/v1', api_key=api_keys["openrouter"], **generation_args),
        
        'local': dspy.LM('ollama_chat/devstral', api_base='http://localhost:11434', **generation_args)
         #  'groq': dspy.LM("groq/llama-3.1-8b-instant", api_key=api_keys["grok_key"], 
      #                  api_base="https://api.groq.com/openai/v1", **generation_args),
    }


def configure_model(model_name='llama'):
    """Configure default DSPy model."""
    if model_name in models:
        dspy.settings.configure(lm=models[model_name])
        return models[model_name]
    raise ValueError(f"Unknown model: {model_name}")


# For thread-local context if needed
thread_local = threading.local()

# Configure root logger
logging.basicConfig(
    level=logging.WARNING,  # Only log INFO or higher from your own app
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# logging.basicConfig(level=logging.ERROR)

# Suppress verbose logs from external libraries
for noisy_logger in ["httpx", "LiteLLM", "uvicorn", "openai", "transformers"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Create your logger
logger = logging.getLogger(__name__)

# Standard ASP rules (unchanged from previous)
standard_rules= """
% standard rules to solve the query based on the facts 
inverse(left, right; right, left; above, below; below, above; front, behind; behind, front;).
is(Y, R2, X) :- is(X, R1, Y), inverse(R1, R2), X != Y.
symmetric(near; far; adjacent;touching).
is(Y, R, X) :- is(X, R, Y), symmetric(R), X != Y.
transitive(above; below; left; right; front; behind).
is(X, R, Z) :- is(X, R, Y), is(Y, R, Z), transitive(R), X != Y, Y != Z, X != Z.
is(O1, R, O2) :- object(O1, _, _, _, B1), object(O2, _, _, _, B2), is(B1, R, B2), O1 != O2, B1 != B2.
is(X, far, Y) :- object(X, _, _, _, B), object(Y, _, _, _, B), not is(X, near, Y), X != Y.
#show query/1.
"""


convert_instructions= u"""
TASK:
Convert a natural language context and question into ASP (Answer Set Programming) facts and a query.

====================
STEP 1 — CONVERT CONTEXT TO FACTS
====================
Extract all blocks, objects, and relationships from the context, sentence by sentence.
Infer implied relations if not explicitly stated.

Allowed facts:
1. Blocks:
   block(a).
   block(a;b;c).   % multiple blocks in one fact

2. Objects:
   object(name, size, color, shape, block).
   - Order is fixed: name, size, color, shape, block
   - Use 'unknown' for unspecified size or color (instead of '_')
   Example:
       object(large_blue_triangle, large, blue, triangle, a).
       object(blue_oval, unknown, blue, oval, b).

3. Relations:
   is(entity1, relation, entity2).
   - Allowed relations: left, right, front, behind, above, below, near, far, adjacent, disconnected
   - Compound relations (e.g., "left and far") must be split into two facts
   - Do NOT use invalid relations like touching, left_far, right_front
   - For "touching edge" cases, infer object-to-object spatial relations instead

Example:
[Sentence]
There is two blocks, A and B. The big blue circle is touching the bottom edge of A, and the green circle is touching the top edge of A. In B, there is a large blue circle near and below a small blue circle. The small blue circle is near and below a large yellow square. There is a medium blue square far to the left of the yellow square.
[Facts]
 block(a;b). object(big_blue_circle, big, green, circle, a). object(green_circle, green, unknown, circle, a). is(green_circle, above, big_blue_circle).object(large_blue_circle, large, blue, circle, b). object(small_blue_circle, small, blue, circle, b). object(large_yellow_square, large, yellow, square, b). object(medium_blue_square, medium, blue, square, b).  is(large_blue_circle, near, small_blue_circle). is(large_blue_circle, below, small_blue_circle). is(small_blue_circle, near, large_yellow_square). is(small_blue_circle, below, large_yellow_square). is(medium_blue_square, far, large_yellow_square).is(medium_blue_square, left, large_yellow_square).
====================
STEP 2 — GENERATE ASP QUERY
====================
Decide query type based on question:

1. YES/NO QUESTIONS:
   query(yes) :- <conditions>.
   query(no) :- not query(yes).

2. WH-QUESTIONS (Relation, Block, Object):
   Use consistent form: query(Answer) :- <conditions>.

RULES:
- Variables (UPPERCASE) must be bound in the body via object/5, is/3, or block/1.
- Wildcards '_' in queries mean "don't care".
- Match specificity from question.
- Variable names must be consistent between query head and body.

====================
STEP 3 — EXAMPLES
====================

--- YES/NO (3 examples) ---
Q: Is the red circle above the white square?
query(yes) :- object(RedCircle, _, red, circle, _),
              object(WhiteSquare, _, white, square, _),
              is(RedCircle, above, WhiteSquare).
query(no) :- not query(yes).

Q: Are all small triangles in the same block?
query(yes) :- #count{B : object(_, small, _, triangle, B)} == 1.
query(no) :- not query(yes).

Q: Is there a blue square far from a yellow triangle?
query(yes) :- object(BS, _, blue, square, _),
              object(YT, _, yellow, triangle, _),
              is(BS, far, YT).
query(no) :- not query(yes).

--- FIND RELATION (3 examples) ---
Q: What is the relation between the red circle and the blue square in block a?
query(Answer) :- object(RedCircle, _, red, circle, a),
                 object(BlueSquare, _, blue, square, a),
                 is(RedCircle, Answer, BlueSquare).

Q: What is the relation between the large triangle in a and the small circle in b?
query(Answer) :- object(LT, large, _, triangle, a),
                 object(SC, small, _, circle, b),
                 is(LT, Answer, SC).

Q: What is the relation between the green square and the red oval?
query(Answer) :- object(GS, _, green, square, _),
                 object(RO, _, red, oval, _),
                 is(GS, Answer, RO).

--- FIND BLOCK (3 examples) ---
Q: Which block has a square to the right of another square?
query(Answer) :- block(Answer),
                 object(Sq1, _, _, square, Answer),
                 object(Sq2, _, _, square, Answer),
                 Sq1 != Sq2,
                 is(Sq1, right, Sq2).

Q: Which block has no large objects?
query(Answer) :- block(Answer),
                 not object(_, large, _, _, Answer).

Q: Which block has a triangle to the left of and right of a blue object?
query(Answer) :- block(Answer),
                 object(Tri, _, _, triangle, Answer),
                 object(B1, _, blue, _, Answer),
                 object(B2, _, blue, _, Answer),
                 B1 != B2,
                 is(Tri, left, B1),
                 is(Tri, right, B2).

--- FIND OBJECT (3 examples) ---
Q: What object is to the left of the large blue thing?
query(Answer) :- object(LBT, large, blue, _, _),
                 is(Answer, left, LBT).

Q: What object is not below the black object in b? the black square or the large yellow square?
query(Answer) :- object(BlackB, _, black, _, b),
                 not is(Answer, below, BlackB).

Q: What object is below the brown thing? the blue rectangle or the small square?
candidate(Obj) :- object(Obj, _, blue, rectangle, _).
candidate(Obj) :- object(Obj, small, _, square, _).
query(Answer) :- object(BrownThing, _, brown, _, _),
                 is(Answer, below, BrownThing),
                 candidate(Answer).

====================
STEP 4 — REVIEW & REFINE
====================
- All facts end with '.'.
- Predicates:
    block/1, object/5, is/3
- Allowed relations only.
- All variables in queries must be bound.
- Facts must match the context exactly.

OUTPUT FORMAT:
% Facts
<facts>
% Query
<query>
"""


refine_instructions= u"""
TASK:
Debug and fix ASP facts and query based on context, question, current ASP code and clingo error messages 

====================
STEP 1 — RE-EXAMINE CONTEXT & QUESTION
====================
1. Determine Question Type:
   - FR (Find Relation): "What is the relation between X and Y?" → query(Answer) :- ...
   - YN (Yes/No): "Does block A have circles?" → query(yes/no) :- ...
   - FB (Find Block): "Which block has triangles?" → query(Answer) :- ...
   - FO (Find Object): "Which object is to the left of the yellow square?" → query(Answer) :- ...

2. Object Inventory:
   - List all objects from the context with full properties

3. Relationship Inventory:
   - List all spatial relations described in the context
   - Include any relations that can be logically inferred from the description

====================
STEP 2 — VALIDATE FACTS COMPLETENESS
====================
Facts must follow strict format:

1. **Blocks**
   block(a).
   block(a;b;c).  % multiple in one fact
   - Block names lowercase (a, b, c)

2. **Objects**
   object(name, size, color, shape, block).
   - Maintain exact order of 5 parameters
   - Use "unknown" for unspecified size/color
   - Example:
       object(large_red_circle, large, red, circle, a).
       object(medium_triangle, medium, unknown, triangle, c).

3. **Relations**
   is(entity1, relation, entity2).
   - Allowed relations only:
     left, right, front, behind, above, below, near, far, adjacent, disconnected
   - One relation per fact

Completeness checklist:
- Every object in context appears in object/5
- Every relationship in context or inferred appears in is/3
- Object names unique & descriptive
- All blocks declared

====================
STEP 3 — ERROR-SPECIFIC FIXES
====================

--- Unsafe Variables ---
Problem: Variables in query are not bound
Fix: Ensure all variables in query head also appear in the body via object/5, is/3, or block/1

--- Syntax / Parsing Error ---
Common issues:
- Missing periods (.) at end of facts/rules/query
- Mismatched parentheses
- Invalid predicate names (letters, numbers, underscores only)
- Wrong argument order in object/5
Fix: Correct syntax and ordering

--- Satisfiable but No Results ---
Problem: Query logic does not match facts
Fix:
1. Verify exact object name matches
2. Ensure required is/3 relations exist
3. Check if inverse relations are missing
4. Align query with question meaning
5. Correct typos

--- Grounding Error ---
Problem: Invalid terms or predicates
Fix:
- Valid predicate names only
- No invalid characters in object names
- Correct predicate arity (block/1, object/5, is/3)

--- Unsatisfiable ---
Problem: Conditions are impossible
Fix:
- Check for contradictory logic
- Ensure facts support query expectations
- Reinterpret question if necessary

====================
STEP 4 — FINAL CHECK
====================
- All variables in queries are bound
- All facts end with '.'
- Only allowed relations used
- Facts match context exactly
- Remove any comments or code block markers in final output, otherwise the code will not be runnable at all. 

====================
OUTPUT FORMAT:
====================
% Facts
<facts>
% Query
<query>
% Rules
<Rules>
#show query/1.
"""

