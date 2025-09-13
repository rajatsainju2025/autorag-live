from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json


class LLMJudge(ABC):
    """Abstract base class for LLM-based judges."""
    
    @abstractmethod
    def judge_answer_relevance(self, question: str, answer: str, context: str) -> float:
        """
        Judge the relevance of an answer to a question given context.
        
        Returns:
            Score between 0.0 and 1.0, where 1.0 is perfectly relevant.
        """
        pass
    
    @abstractmethod
    def judge_faithfulness(self, answer: str, context: str) -> float:
        """
        Judge how faithful an answer is to the provided context.
        
        Returns:
            Score between 0.0 and 1.0, where 1.0 is perfectly faithful.
        """
        pass


class DeterministicJudge(LLMJudge):
    """Simple deterministic judge using heuristics (no LLM required)."""
    
    def judge_answer_relevance(self, question: str, answer: str, context: str) -> float:
        """Simple relevance scoring based on token overlap."""
        q_tokens = set(question.lower().split())
        a_tokens = set(answer.lower().split())
        c_tokens = set(context.lower().split())
        
        # Answer should overlap with both question and context
        q_overlap = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0.0
        c_overlap = len(c_tokens & a_tokens) / len(c_tokens) if c_tokens else 0.0
        
        return min(1.0, (q_overlap + c_overlap) / 2)
    
    def judge_faithfulness(self, answer: str, context: str) -> float:
        """Simple faithfulness scoring based on answer being subset of context."""
        a_tokens = set(answer.lower().split())
        c_tokens = set(context.lower().split())
        
        if not a_tokens:
            return 1.0
            
        overlap_ratio = len(a_tokens & c_tokens) / len(a_tokens)
        return min(1.0, overlap_ratio * 1.2)  # Slight boost for good overlap


class OpenAIJudge(LLMJudge):
    """OpenAI-based judge using GPT models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None
        
        if self.api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                print("OpenAI package not installed. Install with: pip install openai")
    
    def judge_answer_relevance(self, question: str, answer: str, context: str) -> float:
        if not self._client:
            # Fallback to deterministic judge
            det_judge = DeterministicJudge()
            return det_judge.judge_answer_relevance(question, answer, context)
        
        prompt = f"""
        Rate the relevance of this answer to the question on a scale of 0-10.
        
        Question: {question}
        Answer: {answer}
        Context: {context}
        
        Consider:
        - Does the answer address the question?
        - Is the answer supported by the context?
        - How directly does it answer the question?
        
        Return only a number between 0 and 10.
        """
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text) / 10.0
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback
            det_judge = DeterministicJudge()
            return det_judge.judge_answer_relevance(question, answer, context)
    
    def judge_faithfulness(self, answer: str, context: str) -> float:
        if not self._client:
            det_judge = DeterministicJudge()
            return det_judge.judge_faithfulness(answer, context)
        
        prompt = f"""
        Rate how faithful this answer is to the context on a scale of 0-10.
        
        Answer: {answer}
        Context: {context}
        
        Consider:
        - Does the answer contain information not in the context?
        - Is the answer consistent with the context?
        - Does the answer accurately reflect what's stated in the context?
        
        Return only a number between 0 and 10.
        """
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text) / 10.0
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"OpenAI API error: {e}")
            det_judge = DeterministicJudge()
            return det_judge.judge_faithfulness(answer, context)


def get_judge(judge_type: str = "deterministic", **kwargs) -> LLMJudge:
    """Factory function to get a judge instance."""
    if judge_type == "openai":
        return OpenAIJudge(**kwargs)
    elif judge_type == "deterministic":
        return DeterministicJudge()
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")


def save_judge_config(judge: LLMJudge, path: str = "judge_config.json") -> None:
    """Save judge configuration."""
    config = {
        "type": judge.__class__.__name__,
        "model": getattr(judge, 'model', None),
        "version": 1
    }
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_judge_config(path: str = "judge_config.json") -> Dict[str, Any]:
    """Load judge configuration."""
    if not os.path.exists(path):
        return {"type": "deterministic"}
    
    with open(path, 'r') as f:
        return json.load(f)