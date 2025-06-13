# Mathematical/Simulation Model: Formulate or extend the existing model from Workshop #1 to include non-linear or time-dependent factors.
In the first workshop we talked about DQN, and how it could be useful for us to improve the capabilities of the agent and get better result over time. Now, we propose a math model for the agent behaviour similar to a Bellman equation, but for now we think in a equation that approximates to the idea of Bellman equation.
The first proposal is this:

On the other hand, we think that to get a better performance of the agent behaviour with DQN, is a good to make the implementation in a discrete simulated portion of a city.

# Phase Portraits or Diagrams: Illustrate how the agent’s state space unfolds with varying inputs, highlighting attractors or chaotic regimes

The attractors in our case can be:
- A optimal trajectory, because the the agent could converge in one route and not exploring other options, which can lead to a worst behavior most of the times.
- Another could be a loop route, and therefore not it can not reach the destiny.

For now, we have found that traffic is a chaoric regime, because the action of a car stoping could slow the overall speed average, and as many of traffic actors actions are random, then the traffic becomes chaoric.

# Bibliography
- https://huggingface.co/learn/deep-rl-course/en/unit2/bellman-equation
- https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae