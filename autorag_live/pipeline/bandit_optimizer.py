"""Bandit algorithm for online hybrid weight optimization."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass

from autorag_live.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BanditArm:
    """Represents an arm in the multi-armed bandit."""
    weights: Dict[str, float]
    reward_history: List[float]
    pull_count: int

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.reward_history = []
        self.pull_count = 0

    @property
    def average_reward(self) -> float:
        """Calculate average reward for this arm."""
        return float(np.mean(self.reward_history)) if self.reward_history else 0.0

    @property
    def total_reward(self) -> float:
        """Calculate total reward for this arm."""
        return sum(self.reward_history)

    def update(self, reward: float):
        """Update arm with new reward."""
        self.reward_history.append(reward)
        self.pull_count += 1


class UCB1Bandit:
    """Upper Confidence Bound (UCB1) algorithm for multi-armed bandit."""

    def __init__(self, arms: List[BanditArm], exploration_factor: float = 2.0):
        """Initialize UCB1 bandit.

        Args:
            arms: List of bandit arms (weight configurations)
            exploration_factor: Exploration parameter (higher = more exploration)
        """
        self.arms = arms
        self.exploration_factor = exploration_factor
        self.total_pulls = 0

    def select_arm(self) -> BanditArm:
        """Select the best arm using UCB1 formula."""
        if self.total_pulls < len(self.arms):
            # Ensure each arm is pulled at least once
            return self.arms[self.total_pulls]

        # Calculate UCB values for each arm
        ucb_values = []
        for arm in self.arms:
            if arm.pull_count == 0:
                return arm  # Should not happen due to initial pulls

            exploitation = arm.average_reward
            exploration = self.exploration_factor * np.sqrt(
                np.log(self.total_pulls) / arm.pull_count
            )
            ucb_values.append(exploitation + exploration)

        # Return arm with highest UCB value
        best_arm_idx = np.argmax(ucb_values)
        return self.arms[best_arm_idx]

    def update(self, arm: BanditArm, reward: float):
        """Update the selected arm with reward."""
        arm.update(reward)
        self.total_pulls += 1


class ThompsonSamplingBandit:
    """Thompson Sampling algorithm for multi-armed bandit."""

    def __init__(self, arms: List[BanditArm]):
        """Initialize Thompson Sampling bandit.

        Args:
            arms: List of bandit arms (weight configurations)
        """
        self.arms = arms

    def select_arm(self) -> BanditArm:
        """Select arm using Thompson Sampling."""
        # Sample from beta distribution for each arm
        samples = []
        for arm in self.arms:
            if arm.pull_count == 0:
                # Prior: beta(1, 1) for unpulled arms
                sample = np.random.beta(1, 1)
            else:
                # Posterior: beta(successes + 1, failures + 1)
                successes = sum(1 for r in arm.reward_history if r > 0)
                failures = arm.pull_count - successes
                sample = np.random.beta(successes + 1, failures + 1)
            samples.append(sample)

        # Return arm with highest sample
        best_arm_idx = np.argmax(samples)
        return self.arms[best_arm_idx]

    def update(self, arm: BanditArm, reward: float):
        """Update the selected arm with reward."""
        arm.update(reward)


class BanditHybridOptimizer:
    """Bandit-based optimizer for hybrid retriever weights."""

    def __init__(
        self,
        retriever_names: List[str],
        bandit_type: str = "ucb1",
        exploration_factor: float = 2.0,
        num_random_arms: int = 10,
        weight_bounds: Tuple[float, float] = (0.0, 1.0)
    ):
        """Initialize bandit hybrid optimizer.

        Args:
            retriever_names: Names of retrievers to optimize weights for
            bandit_type: Type of bandit algorithm ("ucb1" or "thompson")
            exploration_factor: Exploration parameter for UCB1
            num_random_arms: Number of random weight configurations to generate
            weight_bounds: Min/max bounds for individual weights
        """
        self.retriever_names = retriever_names
        self.bandit_type = bandit_type
        self.exploration_factor = exploration_factor
        self.weight_bounds = weight_bounds

        # Generate initial arms with random weight configurations
        self.arms = self._generate_random_arms(num_random_arms)

        # Initialize bandit algorithm
        if bandit_type == "ucb1":
            self.bandit = UCB1Bandit(self.arms, exploration_factor)
        elif bandit_type == "thompson":
            self.bandit = ThompsonSamplingBandit(self.arms)
        else:
            raise ValueError(f"Unknown bandit type: {bandit_type}")

        logger.info(f"Initialized {bandit_type} bandit with {len(self.arms)} arms")

    def _generate_random_arms(self, num_arms: int) -> List[BanditArm]:
        """Generate random weight configurations."""
        arms = []

        for _ in range(num_arms):
            # Generate random weights that sum to 1
            weights = np.random.dirichlet(np.ones(len(self.retriever_names)))
            weight_dict = {
                name: float(weight)
                for name, weight in zip(self.retriever_names, weights)
            }
            arms.append(BanditArm(weight_dict))

        return arms

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensure weights sum to 1 and are within bounds."""
        total = sum(weights.values())
        if total == 0:
            # Equal weights if all are zero
            normalized = {name: 1.0 / len(weights) for name in weights}
        else:
            normalized = {name: w / total for name, w in weights.items()}

        # Clamp to bounds
        for name in normalized:
            normalized[name] = np.clip(
                normalized[name],
                self.weight_bounds[0],
                self.weight_bounds[1]
            )

        # Renormalize after clamping
        total = sum(normalized.values())
        normalized = {name: w / total for name, w in normalized.items()}

        return normalized

    def suggest_weights(self) -> Dict[str, float]:
        """Suggest next weight configuration to try."""
        arm = self.bandit.select_arm()
        return arm.weights.copy()

    def update_weights(self, weights: Dict[str, float], reward: float):
        """Update bandit with reward for given weight configuration.

        Args:
            weights: Weight configuration that was used
            reward: Reward obtained (higher is better)
        """
        # Normalize weights for comparison
        normalized_weights = self._normalize_weights(weights)

        # Find matching arm
        best_match = None
        best_similarity = -1

        for arm in self.arms:
            # Calculate cosine similarity between weight vectors
            w1 = np.array([normalized_weights.get(name, 0) for name in self.retriever_names])
            w2 = np.array([arm.weights.get(name, 0) for name in self.retriever_names])

            similarity = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = arm

        if best_match is None:
            logger.warning("No matching arm found for weights, creating new arm")
            best_match = BanditArm(normalized_weights)
            self.arms.append(best_match)

        # Update the arm with reward
        self.bandit.update(best_match, reward)

        logger.debug(f"Updated arm with reward {reward:.3f}, similarity {best_similarity:.3f}")

    def get_best_weights(self) -> Dict[str, float]:
        """Get the best weight configuration found so far."""
        if not self.arms:
            return {}

        best_arm = max(self.arms, key=lambda arm: arm.average_reward)
        return best_arm.weights.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the bandit optimization."""
        if not self.arms:
            return {}

        rewards = [arm.average_reward for arm in self.arms]
        pulls = [arm.pull_count for arm in self.arms]

        return {
            "num_arms": len(self.arms),
            "total_pulls": sum(pulls),
            "best_reward": max(rewards) if rewards else 0,
            "average_reward": np.mean(rewards) if rewards else 0,
            "arm_rewards": rewards,
            "arm_pulls": pulls,
            "best_weights": self.get_best_weights()
        }

    def add_custom_arm(self, weights: Dict[str, float]):
        """Add a custom weight configuration as a new arm."""
        normalized_weights = self._normalize_weights(weights)
        new_arm = BanditArm(normalized_weights)
        self.arms.append(new_arm)
        logger.info(f"Added custom arm with weights: {normalized_weights}")


def create_bandit_optimizer(
    retriever_names: List[str],
    bandit_type: str = "ucb1",
    **kwargs
) -> BanditHybridOptimizer:
    """Factory function to create bandit optimizer.

    Args:
        retriever_names: Names of retrievers to optimize
        bandit_type: Type of bandit algorithm
        **kwargs: Additional arguments for BanditHybridOptimizer

    Returns:
        Configured BanditHybridOptimizer instance
    """
    return BanditHybridOptimizer(
        retriever_names=retriever_names,
        bandit_type=bandit_type,
        **kwargs
    )