# Generated answers (llama3.2, notes, top-3 chunks)

- Answers citing the correct lecture: **82.6%**
- Answers citing any lecture: 93.5%
- Correct lecture among the retrieved chunks: 97.8%
- Model said the excerpts do not contain the answer: 0.0%
- Median time per answer: 4.3 s (CPU)

### How is reinforcement learning different from supervised learning?
*Labelled lecture 1 · cited [1]*

Reinforcement learning differs from supervised learning in that there are no labels, only a scalar reward, and feedback is delayed (credit assignment) and delayed by the agent's own actions, which changes the data it sees (exploration vs exploitation).

Source: Lecture 1, Week 1: RL orientation and math refresher

### What is the discounted return and what recursive relation does it satisfy?
*Labelled lecture 1 · cited [1]*

The discounted return is defined as G t = R t+1 + gamma R t+2 + gamma 2 R t+3 + dots = sum k=0 infty gamma k R t+k+1 , with 0 le gamma le 1 . This expression satisfies the recursive relation G t = R t+1 + gamma G t+1 .

Source: Lecture 1, Week 1: RL orientation and math refresher | The RL problem: The RL problem.

### Why doesn't REINFORCE need to know the environment's transition model?
*Labelled lecture 1 · cited [1]*

REINFORCE doesn't need to know the environment's transition model because it only depends on the policy term, which is why it uses the expression ∇θ log p(θ(tau)) = ∑t ∇θ log π(θ(a_t | s_t)), which only involves the policy parameters and the state-action probabilities, but not the transition probabilities. This is a key property of REINFORCE, as stated in the probability refresher from Lecture 1. 

Source: Lecture 1, Probability refresher

### Which identity from probability lets us split an expectation by conditioning on another variable?
*Labelled lecture 1 · cited [1]*

The identity from probability that lets us split an expectation by conditioning on another variable is the Law of Total Expectation. It states that for any random variable X and any random variables Y and Z, E[X] = E[Y[E[X|Y]]]. This is mentioned in the Lecture 1 excerpt on probability refresher.

Source: Lecture 1, Week 1: RL orientation and math refresher | Probability refresher, @ 5

### What does it mean for a process to be Markov?
*Labelled lecture 2 · cited [1]*

A process is Markov if the future state depends only on the current state and not on any of the past states. In other words, the probability of transitioning to a new state is solely determined by the current state and the action taken, and not by the sequence of states and actions that preceded it. This is the Markov property.

Source: Lecture 1, Section

### How can I compute a policy's state values exactly using a matrix inverse?
*Labelled lecture 2 · cited [3]*

To compute a policy's state values exactly using a matrix inverse, you can use the policy evaluation equation from the Policy Iteration section: `v_pi(s) = (I - gamma * P_pi) -1 * r_pi(s)`. This equation involves computing the inverse of a matrix, specifically `(I - gamma * P_pi)`.

Source: Lecture 3, Week 3: Dynamic programming and temporal-difference learning | Policy evaluation ( rl/dp.py )

### Why does repeatedly applying the Bellman backup converge to a unique answer?
*Labelled lecture 2 · cited [2]*

The repeatedly applying the Bellman backup converges to a unique answer because of the gamma-contraction property of the Bellman operator. Specifically, it is shown in the lecture that for any two functions u and v, the distance between their iterates decreases with each iteration, i.e., T pi u - T pi v infty ≤ gamma u - v infty. This property ensures that the Banach fixed-point theorem applies, guaranteeing convergence to the unique fixed point v_pi, which is the optimal value function.

Source: Lecture 2, Week 2

### What per-sweep shrink factor was observed on FrozenLake under a random policy?
*Labelled lecture 2 · cited [2]*

The per-sweep shrink factor observed on FrozenLake under a uniform random policy was less than or equal to gamma = 0.9, which is less than gamma due to some probability mass being absorbed in the holes and goal.

Source: Lecture 2, Week 2: MDPs, Bellman equations and multi-armed bandits

### Why does value iteration still converge when the optimality backup contains a max?
*Labelled lecture 2 · cited [2]*

The Bellman optimality equation contains a max, which makes the operator non-linear. However, it is still a gamma-contraction, meaning that the difference between the max of two functions is greater than or equal to the difference between the two functions themselves, i.e., max f - max g ≥ f - g. This property ensures that value iteration converges.

Source: Lecture 2, Week 2: MDPs, Bellman equations and multi-armed bandits, 'The max makes the operator non-linear, but it is still a gamma-contraction...'

### Which exploration strategy chose the best slot machine most often on the 10-armed testbed?
*Labelled lecture 2 · cited [2]*

The exploration strategy that chose the best slot machine most often on the 10-armed testbed was the Optimistic Q strategy, with a Q value of 0 = 5, with an optimal action rate of 68.9% and a reward of 1.42. 

Source: Lecture 2, Week 2: MDPs, Bellman equations and multi-armed bandits, Multi-armed bandits (rl/bandits.py)

### How does the upper confidence bound rule pick an arm?
*Labelled lecture 2 · cited [2]*

The upper confidence bound (UCB) rule picks an arm by choosing the arm with the maximum value of Q(a) + c*sqrt(ln t / N(a)), where c is a hyperparameter that controls the trade-off between exploration and exploitation.

Source: Lecture 2, Week 2: MDPs, Bellman equations and multi-armed bandits | Multi-armed bandits (rl/bandits.py)

### How do you update a running average estimate of an arm's reward after each pull?
*Labelled lecture 2 · cited [2]*

According to the excerpt from Lecture 2: Week 2: MDPs, Bellman equations and multi-armed bandits | Multi-armed bandits ( rl/bandits.py ), the running average estimate of an arm's reward is updated using the incremental sample average formula: Q n+1 = Q n + 1/n (R n - Q n), where Q n+1 is the new estimate, Q n is the old estimate, R n is the reward obtained on the n-th trial, and n is the number of trials.

Source: Lecture 2, Section: Multi-armed bandits ( rl/bandits.py )

### What are the steps of policy iteration?
*Labelled lecture 3 · cited [3]*

The steps of policy iteration are as follows: 

1. Policy evaluation: compute v π exactly with (I - γ P π) -1 r π. 
2. Policy improvement: π'(s) = argmax a big[r(s,a) + γ sum s' p(s' | s,a), v π(s')big].
3. Repeat steps 1 and 2 until the policy stops changing.

Source: Lecture 3, Week 3: Dynamic programming and temporal-difference learning | Policy iteration (rl/dp.py)

### What does the policy improvement theorem guarantee?
*Labelled lecture 3 · cited none*

The policy improvement theorem guarantees that if q pi(s, pi'(s)) ≥ v pi(s) for all s, then v pi' ≥ v pi.

### How many sweeps did value iteration take compared with policy iteration on the 4x4 lake?
*Labelled lecture 3 · cited [3]*

According to the lecture excerpts, Policy iteration took 3 improvement steps, while Value iteration took 571 sweeps to achieve the same result. 

Source: Lecture 3, Week 3, Value iteration (rl/tabular.py)

### Why does even the optimal agent fail to reach the goal about a quarter of the time on slippery ice?
*Labelled lecture 3 · cited [3]*

The optimal agent fails to reach the goal about a quarter of the time on slippery ice because the intended move happens with probability 1/3 only, not 1, making the exploration-exploitation trade-off challenging. 

Source: Lecture 3, Week 3: Value iteration (rl/tabular.py)

### What is the TD error?
*Labelled lecture 3 · cited [3]*

The TD error is defined as the bracket delta t = R t+1 + gamma V(S t+1 ) - V(S t), where V(S t) is the TD error at time step t. This is used in Temporal-difference learning (TD(0)) to calculate the difference between the estimated value of a state and its true value, which is then used to update the estimate.

Source: Lecture 3: Week 3: Dynamic programming and temporal-difference learning | Temporal-difference learning ( rl/td.py ), Excerpt from Section starting with "TD(0): V(S t) leftarrow V(S t) + alpha, [".

### On the random walk task, which was more accurate, Monte Carlo or TD(0), and by how much?
*Labelled lecture 3 · cited [3]*

According to the random-walk experiment in Lecture 3, Week 3, TD(0) reached 51% lower error than the best Monte-Carlo setting with the same data. Specifically, TD(0) achieved an RMS error of 0.038, which is 51% lower than the Monte-Carlo setting's RMS error of 0.077. 

Source: Lecture 3, Week 3: Temporal-difference learning (rl/td.py)

### What does bootstrapping mean when learning value estimates?
*Labelled lecture 3 · cited none*

Bootstrapping means that the estimate is improved by using its own previous estimate, rather than waiting for the full return of the episode. This reduces variance and allows for online updates. In TD(0), bootstrapping is used by taking the TD error as the bracket delta t = R t+1 + gamma V(S t+1 ) - V(S t), which is the difference between the current state's value and the expected value of the next state.

### What is the difference between the update targets of SARSA and Q-learning?
*Labelled lecture 4 · cited [4]*

The update target of SARSA uses the action actually taken, including epsilon-random ones, so it takes into account the exploration-exploitation trade-off. In contrast, the update target of Q-learning uses the max over next actions, which ignores the chance that exploration pushes it off the edge.

Source: Lecture 4: Week 4: Q-learning and tabular control | Week 4: Q-learning and tabular control

### What conditions on the learning rate are needed for tabular Q-learning to converge?
*Labelled lecture 4 · cited [4]*

Tabular Q-learning converges under the Robbins–Monro conditions, which state that if sum t alpha t = infty and sum t alpha t 2 < infty, and every state-action pair is visited, then the algorithm converges. 

Source: Lecture 4, Week 4: Q-learning and tabular control

### Why does the on-policy agent take a safer route along the cliff?
*Labelled lecture 4 · cited [4]*

SARSA's target uses the action actually taken, including epsilon -random ones, so cells next to the cliff look dangerous and it learns a safer path one row up (-17). This is because SARSA optimises the policy it actually runs, whereas Q-learning ignores the chance of exploration pushing it off the edge.

Source: Lecture 4, Section: Why SARSA walks the long way on CliffWalking

### Why can't we store a lookup table of action values for CartPole or Atari?
*Labelled lecture 4 · cited [5]*

The number of states in CartPole and Atari is too high to be stored in a lookup table. CartPole has 4 continuous numbers as states, while Atari has 84×84×4 pixels as states, resulting in a total of 500 states. This means that storing a lookup table of action values for each state would require a very large table, making it impractical.

Source: Lecture 5, Week 5: Deep Q-Networks | Limitations that motivate DQN

### How well did Q-learning do on the Taxi environment?
*Labelled lecture 4 · cited [4]*

Q-learning did well on the Taxi environment, with a success rate of 0.729 across 3 seeds, and a return of 7.91, which is the highest among the three environments tested, with only a small variation in its performance. 

Source: Lecture 4, Week 4: Q-learning and tabular control | Results (3 seeds, greedy evaluation; scripts/run tabular.py)

### What loss does a deep Q-network minimise?
*Labelled lecture 5 · cited [5]*

A deep Q-network minimizes the squared TD error, specifically the Huber loss, which is defined as L(theta) = E (s,a,r,s') sim D Big[big(underbrace r + gamma (1 - d) max a' Q(s', a'; theta -) y - Q(s, a; theta)big) 2Big].

Source: Lecture 5, Week 5: Deep Q-Networks | From Q-learning to DQN

### Why use the Huber loss and clip gradients in DQN?
*Labelled lecture 5 · cited [5]*

The Huber loss is used instead of squared error because it is more robust to outliers. By using the Huber loss, the model is less affected by large TD errors that may occur due to the target network update. Additionally, the gradient-norm clipping at 10 is used to prevent the update from blowing up due to outlier TD errors. This helps to stabilize the learning process and prevent the model from diverging.

Source: Lecture 5: Week 5: Deep Q-Networks | From Q-learning to DQN: Architecture (rl/dqn.py)

### What is the deadly triad?
*Labelled lecture 5 · cited [5]*

The "deadly triad" refers to the combination of three issues that can cause instability in Deep Q-Networks (DQN): hyperparameter sensitivity, function approximation errors, and off-policy learning. Specifically, the triad consists of issues related to epsilon-decay length, batch size, target-update period, which are discussed in the main README under Week 5 tuning.

Source: Lecture 5: Week 5: Deep Q-Networks | Results on CartPole-v1 @ 00:15:05, Results on CartPole-v1

### Why is storing transitions in a buffer and sampling them randomly helpful?
*Labelled lecture 5 · cited [5]*

Function approximation and bootstrapping can diverge because the strong correlation between consecutive transitions can cause the gradient steps to be non-i.i.d. (Section 11.3). By storing transitions in a buffer and sampling random mini-batches, experience replay breaks this strong correlation, making the gradient steps closer to i.i.d. This helps to stabilize the Q-network. 

Source: Lecture 5, Section 5.2

### How does Double DQN fix overestimation of action values?
*Labelled lecture 5 · cited [5]*

Double DQN fixes overestimation of action values by letting the online network choose the action and the target network evaluate it. The target network is a copy of the Q-network and is refreshed every C steps. This prevents the online network from chasing an outdated target network, which can lead to overestimation of action values. 

Source: Lecture 5, Section 2

### How is the exploration rate scheduled during DQN training?
*Labelled lecture 5 · cited [5]*

There is no mention of the exploration rate in the provided excerpts. This topic is not covered in Lecture 5. You may want to revisit Lecture 5, specifically the section on DQN results on CartPole-v1, to find information on hyperparameter tuning, including the exploration rate.

### What happened to the Q-values when the target network was removed?
*Labelled lecture 5 · cited [5]*

When the target network was removed, the Q-values grew by about 5 orders of magnitude above their theoretical bound, leading to the "deadly triad" effect, where the policy fails.

Source: Lecture 5, Week 5: Deep Q-Networks

### What are the main weaknesses of DQN?
*Labelled lecture 5 · cited [5]*

The main weaknesses of DQN are the discrete action limitations, over-estimation bias, and sample-hungry nature, which are addressed by the two tricks: experience replay and target network. These weaknesses lead to the need for further modifications and extensions, such as Double DQN. Source: Lecture 5, Section: Applications and limitations

### How is the policy gradient theorem derived with the log-derivative trick?
*Labelled lecture 6 · cited [6]*

The policy gradient theorem is derived with the log-derivative trick by first separating the log-probability of a trajectory into a policy part and a dynamics part, and then taking the derivative of the policy part with respect to the policy parameters. This leads to the expression for the policy gradient, which is then applied to the log-derivative trick to eliminate the dynamics terms.

The log-derivative trick replaces the dynamics terms of the log-probability with the expectation of the policy gradient, which is a more manageable expression.

Source: Lecture 6, Week 6: Policy-gradient methods (REINFORCE, A2C, PPO) | Policy-gradient theorem (derivation), 
@ 10:45

### Why does subtracting a state-dependent baseline leave the gradient unbiased?
*Labelled lecture 6 · cited [6]*

According to the lecture, subtracting a state-dependent baseline leaves the gradient unbiased because the baseline, V(s), does not depend on the action, a, and therefore its derivative, b(s), does not contribute to the gradient. As a result, the expected gradient remains unchanged, and the cosine similarity between the two is approximately 1.

Source: Lecture 6, Week 6: Policy-gradient methods (REINFORCE, A2C, PPO) | Baselines: unbiased variance reduction

### By how much did the value baseline reduce the variance of the gradient estimate?
*Labelled lecture 6 · cited [6]*

According to the first excerpt from Lecture 6, Week 6: Policy-gradient methods (REINFORCE, A2C, PPO) | Baselines: unbiased variance reduction, the baseline leaves up to 54% of the variance of the gradient estimate removed.

Source: Lecture 6, Excerpt 1, Week 6

### How does the actor-critic method estimate the advantage?
*Labelled lecture 6 · cited [6]*

The actor-critic method estimates the advantage using the critic's TD error, which is the bootstrapped estimate of r t+1 - V(s t+1 ) + gamma V(s t+1 ) - V(s t) (Excerpt from Lecture 6: Week 6: Policy-gradient methods (REINFORCE, A2C, PPO) | Actor-critic (A2C)).

### What does clipping the probability ratio achieve in PPO?
*Labelled lecture 6 · cited [6]*

Clipping the probability ratio achieves a cheap trust region, so that once the ratio leaves the range [1-epsilon, 1+epsilon] in the direction that would increase the objective, the gradient is zero. This provides a cheap way to prevent large policy steps from collapsing performance.

Source: Lecture 6, Week 6: Policy-gradient methods (REINFORCE, A2C, PPO) | PPO: clipped surrogate objective

### Which algorithm solved LunarLander on every seed?
*Labelled lecture 6 · cited none*

PPO was the only method to solve LunarLander on every seed, with a greedy policy that had a mean of 245 ± 26.

### How can Q-learning style methods handle continuous actions?
*Labelled lecture 7 · cited [7]*

Q-learning style methods, such as SARSA, are typically used for discrete actions, not continuous actions. They learn the value of each state-action pair and update the policy by maximizing the Q-value. DQN and DDPG, on the other hand, are designed for continuous actions and use different methods, such as target networks and Gaussian exploration noise.

Source: Lecture 7, 7:2

### How are the target networks in DDPG updated?
*Labelled lecture 7 · cited [5]*

In DDPG, the target networks are updated using Polyak averaging. Specifically, the target network's parameters θ' are updated as θ' ← τθ' + (1-tau)θ, where τ is a hyperparameter set to 0.005. This means that the target network's parameters are a weighted average of the old and new values, with the old values retaining their influence for a certain number of updates (τ = 0.005). This helps to stabilize the learning process by reducing the effect of sudden changes in the target network. Source: Lecture 5, Section 11.3: 'The Target Network'

### How is a reward model trained from people's pairwise comparisons?
*Labelled lecture 7 · cited [7]*

A reward model is trained from people's pairwise comparisons using the Bradley-Terry reward model. This model estimates the probability of success for one response over another, given the preferences of the humans, and is represented as P(i succ j) = σ(r_i - r_j). The model is fit by maximising the log-likelihood sum log σ(hat r w - hat r l), which is logistic regression on reward differences.

Source: Lecture 7, Week 7: DDPG, RLHF and multi-agent RL | RLHF: Reinforcement Learning from Human Feedback ( rl/rlhf.py ) @ 11:45

### Why add a KL penalty when fine-tuning a language model with RL?
*Labelled lecture 7 · cited [7]*

The KL penalty is added to RLHF to limit "reward hacking", which is the over-exploitation of errors in the reward model. By adding a KL penalty, the policy is encouraged to stay close to the reference model, preventing the model from exploiting errors in the reward function. This helps to stabilize the training process and prevent the model from over-optimizing the reward function. Source: Lecture 7, Section 2

### Why is learning non-stationary when several agents learn at once?
*Labelled lecture 7 · cited [7]*

In multi-agent RL, each agent's environment is non-stationary because the other agents are learning too, and their actions change the data the agent sees. This creates a dynamic environment, where the agent cannot assume that the environment will behave in the same way at different times. The Nash equilibrium concept helps to describe this situation, where no player can gain by changing strategy unilaterally.

Source: Lecture 7, Week 7: DDPG, RLHF and multi-agent RL | Multi-agent RL and game theory 101 (rl/marl.py)

### What do two independent learners end up doing in the prisoner's dilemma?
*Labelled lecture 7 · cited [1]*

Two independent learners, each playing the role of Player 1 in the Prisoner's Dilemma, will defect 95% of the time, even though the cooperative (C, C) strategy pays both players more. This is because they are making a Nash equilibrium decision, where no player can improve their payoff by unilaterally changing their strategy. The exploration rate with epsilon = 0.1 accounts for the remaining 5% of the time, as the learners are still exploring their possible actions.

Source: Lecture 1, Section 1

### Which experiments were run in the final project and with how many seeds?
*Labelled lecture 8 · cited [8]*

The final project experiments include:
- Experiment Script 1: DQN component ablation (run with 3 seeds)
- Experiment Script 2: DQN hyperparameter sensitivity (run with 3 seeds)
- Experiment Script 3: REINFORCE with and without a learned baseline (run with 3 seeds)
Additionally, value-based and policy-based experiments were run on CartPole, including DQN, REINFORCE, A2C, and PPO (run with 3 seeds each), and scaling up experiments on LunarLander and Pendulum, including A2C, PPO, and DDPG (run with 3 seeds each).

Source: Lecture 8, Experiment Script 1 & 2 and Experiment Script 3.

### What were the limitations of the final project's experiments?
*Labelled lecture 8 · cited [8]*

The limitations of the final project's experiments were that they only considered three seeds, which resulted in wide confidence intervals. Additionally, the experiments only performed one-at-a-time hyperparameter sweeps, missing interactions between settings. Furthermore, the same network sizes were used for all algorithms, with no tuning per algorithm beyond standard defaults. 

Source: Lecture 8, Section 3
