import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TimeAwareRetriever:
    """
    Time-Aware Retriever.

    This retriever combines semantic similarity scores with a time decay function.
    It is particularly useful for queries where recency is important (e.g., news,
    financial reports, or rapidly changing documentation). The final score is a
    combination of the semantic score and a time-based weight.
    """

    def __init__(
        self, decay_rate: float = 0.01, time_key: str = "timestamp", score_key: str = "score"
    ):
        """
        Initialize the TimeAwareRetriever.

        Args:
            decay_rate: The rate at which the time weight decays. A higher value means
                        older documents are penalized more heavily.
            time_key: The key in the document metadata containing the timestamp (ISO format or datetime object).
            score_key: The key in the document dictionary containing the initial semantic score.
        """
        self.decay_rate = decay_rate
        self.time_key = time_key
        self.score_key = score_key

    def _calculate_time_weight(self, doc_time: datetime, current_time: datetime) -> float:
        """
        Calculate the time weight using an exponential decay function.

        Args:
            doc_time: The timestamp of the document.
            current_time: The current time (or the time of the query).

        Returns:
            A weight between 0 and 1, where 1 means the document is brand new.
        """
        # Calculate difference in days
        delta = current_time - doc_time
        days_old = max(0, delta.days + delta.seconds / 86400.0)

        # Exponential decay: weight = e^(-decay_rate * days_old)
        weight = math.exp(-self.decay_rate * days_old)
        return weight

    def _parse_timestamp(self, timestamp_val: Any) -> Optional[datetime]:
        """
        Parse a timestamp value into a datetime object.
        """
        if isinstance(timestamp_val, datetime):
            return timestamp_val
        elif isinstance(timestamp_val, str):
            try:
                # Try parsing ISO format
                return datetime.fromisoformat(timestamp_val.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"Could not parse timestamp string: {timestamp_val}")
                return None
        elif isinstance(timestamp_val, (int, float)):
            try:
                # Assume Unix timestamp
                return datetime.fromtimestamp(timestamp_val)
            except Exception:
                logger.warning(f"Could not parse numeric timestamp: {timestamp_val}")
                return None
        return None

    def rerank_by_time(
        self, documents: List[Dict[str, Any]], current_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents by combining their semantic score with a time decay weight.

        Args:
            documents: A list of document dictionaries, each containing a score and metadata.
            current_time: The reference time to calculate age from. Defaults to now.

        Returns:
            The list of documents sorted by their new time-aware score.
        """
        if not documents:
            return []

        if current_time is None:
            current_time = datetime.now()

        for doc in documents:
            # Get the original semantic score
            semantic_score = doc.get(self.score_key, 0.0)

            # Get the document timestamp from metadata
            metadata = doc.get("metadata", {})
            timestamp_val = metadata.get(self.time_key)

            if timestamp_val is not None:
                doc_time = self._parse_timestamp(timestamp_val)
                if doc_time:
                    time_weight = self._calculate_time_weight(doc_time, current_time)
                    # Combine scores (simple multiplication here, could be a weighted sum)
                    doc["time_aware_score"] = semantic_score * time_weight
                    doc["time_weight"] = time_weight
                else:
                    # If parsing failed, fallback to original score
                    doc["time_aware_score"] = semantic_score
                    doc["time_weight"] = 1.0
            else:
                # If no timestamp, assume it's old or don't penalize (depending on use case)
                # Here we don't penalize
                doc["time_aware_score"] = semantic_score
                doc["time_weight"] = 1.0

        # Sort by the new time-aware score
        reranked_docs = sorted(
            documents, key=lambda x: x.get("time_aware_score", 0.0), reverse=True
        )
        return reranked_docs
