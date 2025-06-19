import math
import gym
import random 
import numpy as np
import warnings
import itertools
from typing import List, Optional
from enum import Enum
from datetime import datetime

class LearningTactic(Enum):
    QLEARNING = 1
    SARSA = 2
    HYBRID = 3

class QLearningConfig:
    def __init__(self,
                 environment_name: str = 'CartPole-v1',
                 buckets_sizes: Optional[List[List[float]]] = None,
                 discount_factor: float = 0.99,
                 experiment_rate_max: float = 0.7,
                 experiment_rate_min: float = 0,
                 experiment_rate_decay: float = 0,
                 learning_rate_max: float = 0.7,
                 learning_rate_min: float = 0,
                 learning_rate_decay: float = 0,
                 upper_bounds: Optional[List[float]] = None,
                 lower_bounds: Optional[List[float]] = None):
        
        self.environment_name = environment_name
        self.discount_factor = discount_factor
        self.experiment_rate_max = experiment_rate_max
        self.experiment_rate_min = experiment_rate_min
        self.experiment_rate_decay = experiment_rate_decay
        self.learning_rate_max = learning_rate_max
        self.learning_rate_min = learning_rate_min
        self.learning_rate_decay = learning_rate_decay
        self.buckets_sizes = buckets_sizes
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds


class QLearner:
    def __init__(self, config: QLearningConfig):
        self.environment = gym.make(config.environment_name)
        self.attempt_no = 1

        # Observation space dimensions
        observation_space_dim = self.environment.observation_space.shape[0]
        self.possible_actions = range(self.environment.action_space.n)

        # Flexible default bucket sizes
        default_buckets_sizes = [[1.0 / 2] * 2] * observation_space_dim
        self.buckets_sizes = config.buckets_sizes or default_buckets_sizes

        self.discount_factor = config.discount_factor
        self.experiment_rate_max = config.experiment_rate_max
        self.experiment_rate_min = config.experiment_rate_min
        self.experiment_rate_decay = config.experiment_rate_decay
        self.learning_rate_max = config.learning_rate_max
        self.learning_rate_min = config.learning_rate_min
        self.learning_rate_decay = config.learning_rate_decay
        self.q_dict = {}

        if config.environment_name=='LunarLander-v2':
            self.environment._max_episode_steps = 700

        default_upper_bounds = self.environment.observation_space.high
        default_lower_bounds = self.environment.observation_space.low

        default_upper_bounds = np.where(default_upper_bounds == np.inf, [1] * observation_space_dim, default_upper_bounds)
        default_lower_bounds = np.where(default_lower_bounds == -np.inf, [-1] * observation_space_dim, default_lower_bounds)

        self.upper_bounds = config.upper_bounds or default_upper_bounds
        self.lower_bounds = config.lower_bounds or default_lower_bounds

        self._validate_config()
        self.reset()

    def _validate_config(self):
        """Validate the configuration and provide warnings if needed."""
        for idx, buckets in enumerate(self.buckets_sizes):
            if not math.isclose(sum(buckets), 1.0, rel_tol=1e-5):
                warnings.warn(f"Bucket sizes for dimension {idx+1} do not sum to 1. Current sum: {sum(buckets)}")

        if len(self.upper_bounds) != len(self.lower_bounds):
            warnings.warn("Upper bounds and lower bounds must have the same length.")
        if len(self.upper_bounds) != self.environment.observation_space.shape[0]:
            warnings.warn(f"Upper and lower bounds must match the environment's observation space dimensions.")

        if not (0.0 <= self.discount_factor <= 1.0):
            warnings.warn(f"Discount factor (gamma) {self.discount_factor} should be between 0 and 1.")

        if not (0.0 <= self.experiment_rate_min <= self.experiment_rate_max <= 1.0):
            warnings.warn(f"Exploration rate bounds are not valid: {self.experiment_rate_min} <= epsilon <= {self.experiment_rate_max}. Must be between 0 and 1.")
        if not (0.0 <= self.learning_rate_min <= self.learning_rate_max <= 1.0):
            warnings.warn(f"Learning rate bounds are not valid: {self.learning_rate_min} <= alpha <= {self.learning_rate_max}. Must be between 0 and 1.")

    def reset(self):
        """Reset the Q-learning state and preinitialize q_dict with discretized observation-action pairs."""
        self.attempt_no = 1
        self.q_dict = {}

        bucket_sizes = [len(bucket) for bucket in self.buckets_sizes]

        all_discretized_observations = list(itertools.product(*[range(size) for size in bucket_sizes]))

        for observation in all_discretized_observations:
            self.q_dict[observation] = {a: random.uniform(0.0, 0.01) for a in self.possible_actions}

    def print_parameters(self):
        print("Initial Parameters:")
        print(f"Buckets:    {self.buckets_sizes}")
        print(f"Lower bounds:    {self.lower_bounds}")
        print(f"Upper bounds:    {self.upper_bounds}")
        print(f"Learning rate (alpha) max:   {self.learning_rate_max}")
        print(f"Learning rate (alpha) min:   {self.learning_rate_min}")
        print(f"Learning rate (alpha) decay: {self.learning_rate_decay}")
        print(f"Exploration rate (epsilon) max:   {self.experiment_rate_max}")
        print(f"Exploration rate (epsilon) min:   {self.experiment_rate_min}")
        print(f"Exploration rate (epsilon) decay: {self.experiment_rate_decay}")
        print(f"Discount factor (gamma): {self.discount_factor}")

    def pick_from_knowledge(self, observation: List[int]) -> int:
        """Pick an action based on the Q-values stored in q_dict."""
        return max(self.q_dict[tuple(observation)], key=self.q_dict[tuple(observation)].get)
    
    def pick_action(self, observation):
        """Choose between exploitation and experiment."""
        rand = random.random()
        if rand > self.experiment_rate:
            return self.pick_from_knowledge(observation)
        else:
            return self.environment.action_space.sample()

    def update_knowledge(self, action: int, observation: List[int], new_observation: List[int], reward: float):
        """Update the Q-values using the new structure for q_dict."""
        observation_tuple = tuple(observation)
        new_observation_tuple = tuple(new_observation)

        best_future_q_value = max(self.q_dict[new_observation_tuple].values())
        current_q_value = self.q_dict[observation_tuple][action]
        self.q_dict[observation_tuple][action] = (1 - self.learning_rate) * current_q_value + self.learning_rate * (reward + self.discount_factor * best_future_q_value)

    def update_knowledge_sarsa(self, action: int, new_action: int, observation: List[int], new_observation: List[int], reward: float):
        """Update the Q-values using the SARSA formula."""
        observation_tuple = tuple(observation)
        new_observation_tuple = tuple(new_observation)

        current_q_value = self.q_dict[observation_tuple][action]
        future_q_value = self.q_dict[new_observation_tuple][new_action]
        self.q_dict[observation_tuple][action] = (1 - self.learning_rate) * current_q_value + self.learning_rate * (reward + self.discount_factor * future_q_value)

    def learn(self, max_attempts: int = 10000, tactic: LearningTactic = LearningTactic.HYBRID, switch: float = 0.4, print_stats: bool = False, print_every: int = 100) -> List[float]:
        """Run Q-learning for a number of episodes."""
        rewards = []
        start = datetime.now()
        for episode in range(max_attempts):
            self.experiment_rate = self._calculate_rate(self.experiment_rate_min, self.experiment_rate_max, self.experiment_rate_decay, episode)
            self.learning_rate = self._calculate_rate(self.learning_rate_min, self.learning_rate_max, self.learning_rate_decay, episode)

            if tactic == LearningTactic.SARSA or (tactic == LearningTactic.HYBRID and episode < max_attempts * switch):
                reward_sum = self.attempt_sarsa()
            else:
                reward_sum = self.attempt()

            rewards.append(reward_sum)

            if print_stats and (episode + 1) % print_every == 0:
                print(f"Episode {episode + 1:^10}  | Average from last {print_every}: {np.mean(rewards[-print_every:]):^10.2f}  | Alpha: {self.learning_rate:^10.2f}  | Epsilon: {self.experiment_rate:^10.2f} | Time elapsed: {(datetime.now() - start).total_seconds():^10.2f}")
                start = datetime.now()
        
        return rewards

    def attempt(self) -> float:
        """Perform one episode of training."""
        observation_discrete = self.discretise(self.environment.reset())
        done = False
        reward_sum = 0.0
        while not done:
            action = self.pick_action(observation_discrete)
            new_observation, reward, done, _ = self.environment.step(action)
            new_observation_discrete = self.discretise(new_observation)
            self.update_knowledge(action, observation_discrete, new_observation_discrete, reward)
            observation_discrete = new_observation_discrete
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def attempt_sarsa(self) -> float:
        """Perform one episode of SARSA training."""
        observation = self.discretise(self.environment.reset())
        action = self.pick_action(observation)
        done = False
        reward_sum = 0.0
        while not done:
            new_observation, reward, done, _ = self.environment.step(action)
            new_observation_discrete = self.discretise(new_observation)
            new_action = self.pick_action(new_observation_discrete)
            self.update_knowledge_sarsa(action, new_action, observation, new_observation_discrete, reward)
            action = new_action
            observation = new_observation_discrete
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def _calculate_rate(self, rate_min: float, rate_max: float, decay: float, episode: int) -> float:
        """Calculate the rate for exploration or learning based on the episode."""
        return rate_min + (rate_max - rate_min) * np.exp(-decay * episode)

    def discretise(self, observation: np.ndarray) -> List[int]:
        """Discretize the continuous observation into bucket indices."""
        return [map_to_buckets(observation[i], self.lower_bounds[i], self.upper_bounds[i], self.buckets_sizes[i]) for i in range(len(observation))]
    
    def test(self, sample: int = 100, render: bool = False) -> float:
        """Perform a test of trained model."""
        total_reward = 0.0
        for _ in range(sample):
            reward_sum = 0.0
            observation = self.discretise(self.environment.reset())
            done = False
            while not done:
                if render:
                    self.environment.render()
                
                action = self.pick_from_knowledge(observation)
                new_observation, reward, done, info = self.environment.step(action)
                new_observation_discrete = self.discretise(new_observation)
                observation = new_observation_discrete
                reward_sum += reward
            total_reward += reward_sum
                
        return total_reward / sample

def map_to_buckets(value: float, min_value: float, max_value: float, bucket_sizes: List[float]) -> int:
    """Map a continuous value to a discrete bucket."""
    assert sum(bucket_sizes) == 1, "Bucket sizes must sum to 1."
    assert max_value > min_value, "Max value should be higher than min."
    
    total_range = max_value - min_value
    thresholds = []
    cumulative_size = 0
    for size in bucket_sizes:
        cumulative_size += size
        thresholds.append(min_value + cumulative_size * total_range)
    
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return i
    
    return len(bucket_sizes) - 1
