"""
Knowledge graph package initialization.
"""

from .graph import Entity, KnowledgeGraph, KnowledgeGraphNode, Relation

__all__ = [
    "Entity",
    "Relation",
    "KnowledgeGraphNode",
    "KnowledgeGraph",
]
