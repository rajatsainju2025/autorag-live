from typing import List

def rewrite_query(query: str, num_rewrites: int = 3) -> List[str]:
    """
    Generates paraphrases of a query.
    This is a placeholder for a more sophisticated rule-based or LLM-based implementation.
    """
    rewrites = []
    for i in range(num_rewrites):
        rewrites.append(f"{query} (rewrite {i+1})")
    return rewrites
