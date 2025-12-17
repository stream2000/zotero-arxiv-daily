# prompts.py

PAPER_EVALUATION_PROMPT = """
You are an expert research assistant.

My Previous Papers (Context):
The following is a list of papers I have recently read or saved. Use these to infer my specific research interests (topics, methods, organisms, biological questions).
{zotero_context}

Task: Evaluate the following {batch_size} NEW papers. 
For each paper, assign a relevance score from 0 to 10 based on how well it aligns with the interests inferred from my previous papers.
Note: Some papers may not have an abstract available. In such cases, evaluate relevance based on the Title alone.

Scoring Criteria:
- 10: Critical read. Directly addresses my core interests/methods found in the context.
- 7-9: Highly relevant. Strong connection to my field.
- 4-6: Tangentially relevant. Might have some useful info.
- 0-3: Irrelevant.

Output Format:
Return ONLY a list of lines, one for each paper in order (1 to {batch_size}).
Each line must follow this format:
ID | SCORE | REASON

Where:
- ID is the paper number (1, 2, ...).
- SCORE is an integer (0-10).
- REASON is a very brief explanation (max 10 words).

Example:
1 | 9 | Novel deep learning method for spatial data.
2 | 2 | Focuses on clinical trials, unrelated.

Papers to Evaluate:
{batch_text}
"""
