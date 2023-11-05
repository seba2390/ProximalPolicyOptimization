# ProximalPolicyOptimization

**Proximal Policy Optimization** (PPO) [[1]](#1) is a __Reinforcement Learning__ (RL) algorithm.
Conceptually, RL is a branch of __Machine Learning__ (ML) in which problems as generally modelled as 
the process of some _agent_ learning to make decisions by performing 
actions in an environment, with the intent of maximizing some pre-defined 
notion of reward.

In the realm of RL, PPO is defined as an **on-policy algorithm**, which means that it learns directly from the actions
it takes, as opposed to **off-policy** algorithms like [Q-Learning](https://en.wikipedia.org/wiki/Q-learning), which learns
from actions that a different policy might take.

PPO also belongs to the family of **Policy Gradient methods**, which means that it is also characterized
by trying to maximize the Expected cumulative reward, by means of gradient ascent/descent.

On of the defining and unique aspects of PPO, is that it is designed such that it attempts to take the 
largest possible improvement step on a policy, without causing the performance to collapse (which is a 
common problem with these types of algorithms).

PPO addresses this challenge by utilizing a concept known as [Trust Region Optimization](https://en.wikipedia.org/wiki/Trust_region), but in a more computationally efficient 
and simpler manner, than its predecessor, **Trust Region Policy Optimization** (TRPO) [[2]](#2). 

Specifically, PPO achieves this by limiting the amount of change in the policy at each update, thus 
avoiding too large updates that could lead to performance collapse. It uses a [clipped version of the 
policy gradient objective](https://huggingface.co/learn/deep-rl-course/unit8/clipped-surrogate-objective) to prevent 
too large policy updates. The "clip" in PPO restricts the estimated ratio of the new policy to the old
policy to be within a certain range ( $1\pm \varepsilon$, where epsilon is a small value like 0.1 or 0.2). If the ratio is outside of this range, the objective function is modified to "clip" it back within the range, which prevents the policy from changing too much.

This objective allows for frequent policy updates with reduced sample complexity compared to other methods while still ensuring the mathematical stability of the policy improvement process.

## Installation 
##### MacOS/Unix: 
env/bin/python -m pip install -r requirements.txt

##### Windows: 
env\bin\python -m pip install -r requirements.txt

N.B. See [pip 'freeze' documentation](https://pip.pypa.io/en/stable/cli/pip_freeze/) for detailed explanation.


## References
<a id="1">[1]</a> John Schulman, Filip Wolski, et. al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). Arxiv (v2). <br>
<a id="2">[2]</a> John Schulman, Sergey Levine, et. al. (2017). [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Arxiv (v5).
