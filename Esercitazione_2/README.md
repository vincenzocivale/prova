# Esercitazione 2 - Deep Reinforcement Learning Laboratory

Implementazione completa di algoritmi Deep Reinforcement Learning con architettura modulare avanzata. Questo progetto comprende soluzioni ottimizzate per problemi di controllo discreto (CartPole) e continuo (CarRacing) utilizzando algoritmi policy gradient all'avanguardia.


### **Exercise 1: REINFORCE Vanilla on CartPole**

To address the CartPole environment, a reimplementation of the REINFORCE algorithm was carried out. REINFORCE is an on-policy method that updates the policy directly based on observed trajectories, without relying on value function estimation. This makes it particularly suitable for introducing the fundamental principles of stochastic policy optimization.

The adopted neural network is a simple feed-forward architecture with two hidden layers of 128 neurons each and ReLU activations. The inputs (position, velocity, pole angle, and angular velocity) are mapped to two logits corresponding to the available actions.
Action selection is performed using a categorical distribution derived from the softmax of the logits, from which actions are sampled stochastically. This approach ensures a good balance between exploration and exploitation and aligns more closely with the probabilistic nature of the REINFORCE algorithm compared to deterministic selection.

![Cartpole Reward during Training](images/cartpole_reinforce_training.png)

The graph shows the trend in the reward obtained per episode during training, together with the moving average over 100 episodes. After an initial phase of gradual learning, the agent achieves high performance around episode 120, with rewards stabilising close to the maximum (500).

However, two significant phases of instability are observed, around episodes 200 and 500, indicating a temporary degradation in performance. These fluctuations are typical of the REINFORCE algorithm, which is sensitive to trajectory variance and the absence of a baseline.


### **Exercise 2: REINFORCE with Baseline CartPole**

After implementing the vanilla REINFORCE algorithm, one of its main limitations became evident: the high variance in return estimates. To improve training stability, a **value baseline** was introduced. This addition enables the agent to compute a *relative advantage* rather than relying on absolute returns.

Instead of judging actions solely based on the total reward obtained, the agent now considers how much better (or worse) a particular outcome is compared to what was expected from that state. This shift from absolute to contextual evaluation leads to more robust learning.

The implementation relies on two distinct neural networks:

- **Policy Network**: A feed-forward network with two hidden layers (128 units each), outputting logits over the two possible actions.
- **Value Network**: A similarly structured network that outputs a single scalar value representing the estimated value of the state.

```python
class PolicyNet(nn.Module):
    def __init__(self, n_states=4, n_actions=2, n_hidden=128):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.action_head = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.action_head(x)

class ValueNet(nn.Module):
    def __init__(self, n_states=4, n_hidden=128):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.value_head = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value_head(x)
```

Separate networks were preferred over a shared architecture with two heads, to reduce interference between action selection and state evaluation,  two distinct cognitive tasks that benefit from independent representations.
Learning Mechanism: Advantage Estimation

The policy update is based on the advantage function, computed as:

A(st,at)=Gt−V(st)


Where Gt​ is the total return from time step t, and V(st​) is the value estimated by the value network.
Using this advantage rather than raw returns leads to lower variance and faster convergence.


| Vanilla REINFORCE | REINFORCE con Baseline |
|:-----------------:|:---------------------:|
| ![Vanilla REINFORCE](images/cartpole_reinforce_training.png) | ![REINFORCE Baseline](images/cartpole_reinforce_training.png.png) |


As shown in the plots, the performance difference between the two approaches is :

- **Vanilla REINFORCE** shows high variance in both per-episode rewards and the moving average, requiring more episodes to reach the maximum reward. Learning is noisy and often unstable, with frequent dips and slow improvement.
  
- **REINFORCE with Value Baseline**, on the other hand, displays significantly **faster convergence** and **greater stability**. The moving average increases more smoothly and reaches the performance ceiling (reward = 500) in fewer episodes. The value baseline helps reduce variance by centering the updates around state-specific expectations, resulting in a more efficient learning process.

While both methods eventually achieve high rewards, the baseline-enhanced version learns **faster**, **more reliably**, and **recovers better** from performance drops.
