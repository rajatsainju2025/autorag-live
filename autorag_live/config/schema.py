try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - pydantic optional in some dev setups
    BaseModel = object

    def Field(*a, **k):
        return None


class PipelineConfigSchema(BaseModel):
    """Pydantic schema for pipeline configuration.

    Keeps a small, explicit schema that can be used for validation and
    configuration migration tooling.
    """

    enable_routing: bool = Field(True, description="Enable routing stage")
    enable_knowledge_graph: bool = Field(True, description="Enable KG stage")
    enable_multi_agent: bool = Field(True, description="Enable multi-agent stage")
    enable_safety: bool = Field(True, description="Enable safety checks")
    enable_evaluation: bool = Field(False, description="Enable evaluation")

    max_retrieval_results: int = Field(10, description="Number of retrieval results to fetch")
    multi_agent_timeout: float = Field(30.0, description="Timeout for multi-agent collaboration")

    class Config:  # pydantic config
        extra = "forbid"


__all__ = ["PipelineConfigSchema"]
