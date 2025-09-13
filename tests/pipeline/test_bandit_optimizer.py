"""Tests for bandit optimizer."""

import pytest
import numpy as np
from unittest.mock import patch

from autorag_live.pipeline.bandit_optimizer import (
    BanditArm,
    UCB1Bandit,
    ThompsonSamplingBandit,
    BanditHybridOptimizer,
    create_bandit_optimizer
)


class TestBanditArm:
    """Test BanditArm functionality."""

    def test_bandit_arm_initialization(self):
        """Test bandit arm initialization."""
        weights = {"bm25": 0.6, "dense": 0.4}
        arm = BanditArm(weights)

        assert arm.weights == weights
        assert arm.reward_history == []
        assert arm.pull_count == 0
        assert arm.average_reward == 0.0
        assert arm.total_reward == 0.0

    def test_bandit_arm_update(self):
        """Test updating bandit arm with rewards."""
        arm = BanditArm({"bm25": 0.5, "dense": 0.5})

        arm.update(0.8)
        assert arm.reward_history == [0.8]
        assert arm.pull_count == 1
        assert arm.average_reward == 0.8
        assert arm.total_reward == 0.8

        arm.update(0.6)
        assert arm.reward_history == [0.8, 0.6]
        assert arm.pull_count == 2
        assert arm.average_reward == 0.7
        assert arm.total_reward == 1.4


class TestUCB1Bandit:
    """Test UCB1 bandit algorithm."""

    def test_ucb1_initialization(self):
        """Test UCB1 bandit initialization."""
        arms = [BanditArm({"bm25": 0.5, "dense": 0.5}) for _ in range(3)]
        bandit = UCB1Bandit(arms, exploration_factor=1.5)

        assert bandit.arms == arms
        assert bandit.exploration_factor == 1.5
        assert bandit.total_pulls == 0

    def test_ucb1_initial_arm_selection(self):
        """Test that UCB1 selects each arm at least once initially."""
        arms = [BanditArm({"bm25": 0.5, "dense": 0.5}) for _ in range(3)]
        bandit = UCB1Bandit(arms)

        # First selections should be sequential
        assert bandit.select_arm() == arms[0]
        bandit.update(arms[0], 0.8)
        assert bandit.total_pulls == 1

        assert bandit.select_arm() == arms[1]
        bandit.update(arms[1], 0.6)
        assert bandit.total_pulls == 2

        assert bandit.select_arm() == arms[2]
        bandit.update(arms[2], 0.9)
        assert bandit.total_pulls == 3

    def test_ucb1_ucb_calculation(self):
        """Test UCB value calculation after initial pulls."""
        arms = [BanditArm({"bm25": 0.5, "dense": 0.5}) for _ in range(2)]
        bandit = UCB1Bandit(arms)

        # Initial pulls
        bandit.select_arm()
        bandit.update(arms[0], 1.0)
        bandit.select_arm()
        bandit.update(arms[1], 0.5)

        # Next selection should use UCB
        selected = bandit.select_arm()
        # Should select arm with higher UCB value
        assert selected in arms


class TestThompsonSamplingBandit:
    """Test Thompson Sampling bandit algorithm."""

    @patch('numpy.random.beta')
    def test_thompson_sampling_selection(self, mock_beta):
        """Test Thompson Sampling arm selection."""
        # Mock beta distribution samples
        mock_beta.side_effect = [0.3, 0.8]  # Second arm has higher sample

        arms = [BanditArm({"bm25": 0.5, "dense": 0.5}) for _ in range(2)]
        bandit = ThompsonSamplingBandit(arms)

        selected = bandit.select_arm()
        assert selected == arms[1]  # Should select arm with higher sample

        # Verify beta was called for each arm
        assert mock_beta.call_count == 2

    def test_thompson_sampling_update(self):
        """Test Thompson Sampling update."""
        arms = [BanditArm({"bm25": 0.5, "dense": 0.5})]
        bandit = ThompsonSamplingBandit(arms)

        bandit.update(arms[0], 0.8)
        assert arms[0].reward_history == [0.8]
        assert arms[0].pull_count == 1


class TestBanditHybridOptimizer:
    """Test BanditHybridOptimizer functionality."""

    def test_optimizer_initialization(self):
        """Test bandit optimizer initialization."""
        retriever_names = ["bm25", "dense", "hybrid"]
        optimizer = BanditHybridOptimizer(
            retriever_names=retriever_names,
            bandit_type="ucb1",
            num_random_arms=5
        )

        assert optimizer.retriever_names == retriever_names
        assert optimizer.bandit_type == "ucb1"
        assert len(optimizer.arms) == 5

        # Check that weights sum to approximately 1
        for arm in optimizer.arms:
            total_weight = sum(arm.weights.values())
            assert abs(total_weight - 1.0) < 1e-6

    def test_weight_normalization(self):
        """Test weight normalization."""
        optimizer = BanditHybridOptimizer(["bm25", "dense"])

        # Test normalization of weights that don't sum to 1
        weights = {"bm25": 2.0, "dense": 1.0}
        normalized = optimizer._normalize_weights(weights)

        assert abs(normalized["bm25"] - 2/3) < 1e-6
        assert abs(normalized["dense"] - 1/3) < 1e-6

    def test_suggest_weights(self):
        """Test weight suggestion."""
        optimizer = BanditHybridOptimizer(["bm25", "dense"], num_random_arms=3)

        weights = optimizer.suggest_weights()
        assert isinstance(weights, dict)
        assert set(weights.keys()) == {"bm25", "dense"}
        assert all(isinstance(w, float) for w in weights.values())

    def test_update_weights(self):
        """Test updating weights with reward."""
        optimizer = BanditHybridOptimizer(["bm25", "dense"], num_random_arms=2)

        # Get initial weights
        weights = optimizer.suggest_weights()

        # Update with reward
        optimizer.update_weights(weights, 0.8)

        # Check that some arm was updated
        updated_arms = [arm for arm in optimizer.arms if arm.pull_count > 0]
        assert len(updated_arms) > 0

    def test_get_best_weights(self):
        """Test getting best weights."""
        optimizer = BanditHybridOptimizer(["bm25", "dense"], num_random_arms=3)

        # Manually set rewards for testing
        optimizer.arms[0].update(0.9)
        optimizer.arms[1].update(0.7)
        optimizer.arms[2].update(0.8)

        best_weights = optimizer.get_best_weights()
        assert isinstance(best_weights, dict)
        assert set(best_weights.keys()) == {"bm25", "dense"}

    def test_get_statistics(self):
        """Test getting optimizer statistics."""
        optimizer = BanditHybridOptimizer(["bm25", "dense"], num_random_arms=2)

        # Add some data
        optimizer.arms[0].update(0.8)
        optimizer.arms[1].update(0.6)

        stats = optimizer.get_statistics()

        assert stats["num_arms"] == 2
        assert stats["total_pulls"] == 2
        assert stats["best_reward"] == 0.8
        assert "best_weights" in stats

    def test_add_custom_arm(self):
        """Test adding custom arm."""
        optimizer = BanditHybridOptimizer(["bm25", "dense"], num_random_arms=1)

        initial_num_arms = len(optimizer.arms)

        custom_weights = {"bm25": 0.7, "dense": 0.3}
        optimizer.add_custom_arm(custom_weights)

        assert len(optimizer.arms) == initial_num_arms + 1

        # Check that weights were normalized
        new_arm = optimizer.arms[-1]
        total_weight = sum(new_arm.weights.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_thompson_sampling_optimizer(self):
        """Test optimizer with Thompson Sampling."""
        optimizer = BanditHybridOptimizer(
            ["bm25", "dense"],
            bandit_type="thompson",
            num_random_arms=3
        )

        assert isinstance(optimizer.bandit, ThompsonSamplingBandit)

        # Test basic functionality
        weights = optimizer.suggest_weights()
        assert isinstance(weights, dict)

    def test_invalid_bandit_type(self):
        """Test invalid bandit type raises error."""
        with pytest.raises(ValueError, match="Unknown bandit type"):
            BanditHybridOptimizer(
                ["bm25", "dense"],
                bandit_type="invalid",
                num_random_arms=3
            )


class TestCreateBanditOptimizer:
    """Test bandit optimizer factory function."""

    def test_create_ucb1_optimizer(self):
        """Test creating UCB1 optimizer."""
        optimizer = create_bandit_optimizer(
            retriever_names=["bm25", "dense"],
            bandit_type="ucb1",
            num_random_arms=5
        )

        assert isinstance(optimizer, BanditHybridOptimizer)
        assert optimizer.bandit_type == "ucb1"

    def test_create_thompson_optimizer(self):
        """Test creating Thompson Sampling optimizer."""
        optimizer = create_bandit_optimizer(
            retriever_names=["bm25", "dense"],
            bandit_type="thompson",
            num_random_arms=3
        )

        assert isinstance(optimizer, BanditHybridOptimizer)
        assert optimizer.bandit_type == "thompson"