# Deep Reinforcement Learning Laboratory

## Overview

This repository contains implementations and experiments for Deep Reinforcement Learning algorithms, focusing on policy gradient methods. The project explores REINFORCE algorithm variants and their application to different environments, from discrete control tasks to continuous visual control.

### Project Structure

```
Esercitazione_2/
├── src/                           # Source code modules
│   ├── cartpole/                 # CartPole environment implementations
│   │   ├── models.py            # Policy and Value network architectures
│   │   ├── trainers.py          # REINFORCE training algorithms
│   │   ├── config.py            # Training configurations
│   │   └── utils.py             # Utility functions
│   ├── CarRacing/               # Car Racing environment implementations
│   │   ├── net.py               # CNN architectures for visual input
│   │   ├── agent.py             # PPO agent implementation
│   │   ├── environment.py       # Environment wrapper
│   │   ├── hybrid_preprocessor.py # Geometric feature extraction
│   │   └── train.py             # Training pipeline
│   └── common/                  # Shared utilities
├── model/                       # Saved model checkpoints
├── record/                      # Training videos
├── images/                      # Training plots and results
├── Cartpole.ipynb             # CartPole experiments notebook
├── DLA_Lab2_DRL.ipynb         # Main experiments notebook
└── README.md                   # This documentation
```


## Exercises Overview

### Table of Contents

- [Exercise 1: REINFORCE Vanilla on CartPole](#exercise-1-reinforce-vanilla-on-cartpole)
- [Exercise 2: REINFORCE with Baseline CartPole](#exercise-2-reinforce-with-baseline-cartpole)
- [Exercise 3.2: REINFORCE on Lunar Lander](#exercise-32-reinforce-on-lunar-lander)
- [Exercise 3: Car Racing OpenAI](#exercise-3-car-racing-openai)

### **Exercise 1: REINFORCE Vanilla on CartPole**

To address the CartPole environment, a reimplementation of the REINFORCE algorithm was carried out. REINFORCE is an on-policy method that updates the policy directly based on observed trajectories, without relying on value function estimation. This makes it particularly suitable for introducing the fundamental principles of stochastic policy optimization.

The adopted neural network is implemented in [`src/cartpole/models.py`](src/cartpole/models.py) as the [`Policy`](src/cartpole/models.py) class - a simple feed-forward architecture with two hidden layers of 128 neurons each and ReLU activations. The inputs (position, velocity, pole angle, and angular velocity) are mapped to two logits corresponding to the available actions.

Action selection is performed using a categorical distribution derived from the softmax of the logits, from which actions are sampled stochastically. This approach ensures a good balance between exploration and exploitation and aligns more closely with the probabilistic nature of the REINFORCE algorithm compared to deterministic selection.

![REINFORCE Baseline](images/cartpole_baseline.png)

The graph shows the trend in the reward obtained per episode during training, together with the moving average over 100 episodes. After an initial phase of gradual learning, the agent achieves high performance around episode 120, with rewards stabilising close to the maximum (500).

However, two significant phases of instability are observed, around episodes 200 and 500, indicating a temporary degradation in performance. These fluctuations are typical of the REINFORCE algorithm, which is sensitive to trajectory variance and the absence of a baseline.

[Cart_Pole.webm](https://github.com/user-attachments/assets/b4b65870-ba21-4e0b-96ad-fb1441740fa9)


### **Exercise 2: REINFORCE with Baseline CartPole**

After implementing the vanilla REINFORCE algorithm, one of its main limitations became evident: the high variance in return estimates. To improve training stability, a **value baseline** was introduced. This addition enables the agent to compute a *relative advantage* rather than relying on absolute returns.

Instead of judging actions solely based on the total reward obtained, the agent now considers how much better (or worse) a particular outcome is compared to what was expected from that state. This shift from absolute to contextual evaluation leads to more robust learning.

The implementation relies on two distinct neural networks defined in [`src/cartpole/models.py`](src/cartpole/models.py):

- **[`Policy`](src/cartpole/models.py)**: A feed-forward network with two hidden layers (128 units each), outputting logits over the two possible actions.
- **[`ValueNetwork`](src/cartpole/models.py)**: A similarly structured network that outputs a single scalar value representing the estimated value of the state.

```python
# Network architectures from src/cartpole/models.py

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
```

Separate networks were preferred over a shared architecture with two heads, to reduce interference between action selection and state evaluation, two distinct cognitive tasks that benefit from independent representations.

#### Learning Mechanism: Advantage Estimation

The policy update is based on the advantage function, computed as:

**A(s_t, a_t) = G_t - V(s_t)**

Where G_t is the total return from time step t, and V(s_t) is the value estimated by the value network.
Using this advantage rather than raw returns leads to lower variance and faster convergence.

The complete baseline implementation is available in [`src/cartpole/trainers.py`](src/cartpole/trainers.py) with the enhanced REINFORCE algorithm.


| Vanilla REINFORCE | REINFORCE con Baseline |
|:-----------------:|:---------------------:|
| ![Vanilla REINFORCE](images/cartpole_reinforce_training.png) | ![REINFORCE Baseline](images/cartpole_baseline.png) |


As shown in the plots, the performance difference between the two approaches:

- **Vanilla REINFORCE** shows high variance in both per-episode rewards and the moving average, requiring more episodes to reach the maximum reward. Learning is noisy and often unstable, with frequent dips and slow improvement.
  
- **REINFORCE with Value Baseline**, on the other hand, displays significantly **faster convergence** and **greater stability**. The moving average increases more smoothly and reaches the performance ceiling (reward = 500) in fewer episodes. The value baseline helps reduce variance by centering the updates around state-specific expectations, resulting in a more efficient learning process.

While both methods eventually achieve high rewards, the baseline-enhanced version learns **faster**, **more reliably**, and **recovers better** from performance drops.


### **Exercise 3.2: REINFORCE on Lunar Lander**

Following the successful implementation of REINFORCE on CartPole, the next challenge involved applying the same algorithms to the more complex **LunarLander-v3** environment. Unlike CartPole's simple pole-balancing task, Lunar Lander requires the agent to learn sophisticated control strategies for safely landing a spacecraft while managing fuel consumption and avoiding crashes.

The Lunar Lander environment presents a continuous 8-dimensional state space including position, velocity, orientation, and contact sensors, with four discrete actions: do nothing, fire left engine, fire main engine, and fire right engine. The reward structure incentivizes smooth landings between the flags (+100 to +140 points) while penalizing crashes (-100 points) and fuel consumption.


The experimental design maintained consistency with the CartPole implementation while adapting hyperparameters for the increased complexity:

- **Network Architecture**: Both policy and value networks utilized 64 hidden units (increased from 16 for CartPole) to handle the higher-dimensional state space
- **Training Episodes**: Extended to 2,500 episodes to account for the environment's complexity
- **Learning Rate**: Reduced to 5×10⁻³ for more stable learning
- **Discount Factor**: Set to γ = 0.99 to properly value long-term rewards in the landing sequence

The same dual-network approach was employed, with separate policy and value networks to enable REINFORCE with baseline training while maintaining clear separation of concerns between action selection and state evaluation.

The comparative analysis between vanilla REINFORCE and REINFORCE with baseline revealed significant performance differences in the Lunar Lander environment:

[Lunar_Lander REINFORCE with Baseline Result](https://github.com/user-attachments/assets/a04095c2-a592-4253-bc42-2fad203713fb)


**Quantitative Performance Metrics:**

| Method | Final Training Score | Mean Evaluation Performance | Standard Deviation |
|--------|---------------------|---------------------------|-------------------|
| **REINFORCE (Vanilla)** | -45.28 | 62.61 | ±146.27 |
| **REINFORCE with Baseline** | **242.03** | **201.26** | **±79.30** |


These results corroborate established theoretical findings in policy gradient methods, specifically demonstrating that **variance reduction techniques become increasingly critical as environment complexity increases**. The Lunar Lander environment's continuous dynamics, sparse rewards, and safety constraints amplify the high-variance problem inherent in vanilla REINFORCE, making baseline methods not just beneficial but essential for successful learning.

The substantial performance gap observed here (287 points) exceeds the improvements seen in simpler environments like CartPole, suggesting that **the benefits of variance reduction scale with task complexity**. This finding has important implications for the practical deployment of policy gradient methods in real-world applications, where environmental complexity and safety requirements make consistent, low-variance learning paramount.


### **Exercise 3: Car Racing OpenAI**

After achieving good results with REINFORCE on CartPole, the transition to CarRacing-v3 represented a significant leap in terms of complexity. Unlike CartPole, where the agent observes a low-dimensional vector state, CarRacing requires the processing of 96×96×3 RGB images to make real-time driving decisions.

The agent was trained on a discretised action space consisting of five macro-actions (no action, left/right turn, acceleration, braking). This choice, inspired by suggestions from [notanymike](https://notanymike.github.io/Solving-CarRacing/), stabilised training, avoiding the noisy updates and divergence observed with continuous actions.

#### Architecture and Implementation

The architecture is implemented in [`src/CarRacing/net.py`](src/CarRacing/net.py) as a convolutional neural network with three layers (16 to 64 filters, decreasing kernels), followed by two separate output heads for policy and value estimation, sharing a backbone to extract common visual features.

Key components of the CarRacing implementation:

- **[`CarRacingNet`](src/CarRacing/net.py)**: CNN architecture for visual processing
- **[`PPOAgent`](src/CarRacing/agent.py)**: PPO agent with clipped objective function
- **[`HybridPreprocessor`](src/CarRacing/hybrid_preprocessor.py)**: Geometric feature extraction module
- **[`CarRacingEnvironment`](src/CarRacing/environment.py)**: Environment wrapper 
- **[`train_ppo`](src/CarRacing/train.py)**: Complete training pipeline

To improve spatial reasoning, a geometric preprocessing module ([`HybridPreprocessor`](src/CarRacing/hybrid_preprocessor.py)) was added that extracts features such as curvature, proximity to edges, and optimal driving direction, providing structured signals that facilitate learning. Furthermore, to handle temporal dependence, the state includes the last four actions performed (action stacking), introducing a temporal memory that improves control continuity.

Finally, PPO's “clipped” objective function ensured stable and controlled policy updates, preventing catastrophic forgetting and promoting effective convergence despite the high dimensionality of the visual input and the complexity of the task.

![car_racing_episode_reward](https://github.com/user-attachments/assets/4b8a548e-9241-4990-a229-b3e6a3f4394d)

Despite various attempts, the results obtained in CarRacing-v3 were disappointing overall. As shown in the video, the agent proceeds very slowly while remaining on the road, highlighting a conservative policy that limits the reward.

[carracing_test_current.webm](https://github.com/user-attachments/assets/3602e2bd-c9cd-4c07-bb62-f5b944b837ea)


In my opinion, the main reason why the system does not work well on CarRacing-v3 is related to the complexity of the environment compared to that encountered in previous exercises such as CartPole. In CartPole, the agent works with a simple vector state, whereas here it has to interpret 96×96 RGB images, so the input space is much larger and more difficult to manage. The convolutional network I used has a fairly simple structure, and is probably not sufficient to extract meaningful visual features to drive effectively.

Furthermore, the decision to discretise the action space into five macro-actions to simplify learning had a side effect: the agent is unable to modulate acceleration or steering well, and ends up adopting a very cautious policy, staying on the track but going very slowly, thus earning low rewards.

Another possible cause of the system's failure is related to the choice of reward function, in particular the penalty assigned when the car goes off the track. I set a rather severe penalty to discourage bad behaviour, but I realise that this punishment may be too high compared to the reward obtained simply for proceeding correctly along the route.

The result is that the agent tends to avoid any risk, even at the cost of slowing down excessively or not making the most of the track, thus developing a very conservative policy. In practice, the system prefers “not to make mistakes” rather than “doing well”, and this leads to inefficient behaviour that penalises the overall reward.


