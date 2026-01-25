# Math Foundation RL

### Concepts Memo

Discount Rate $\gamma$ :

- $\gamma$ is close to 0 $\rightarrow$ the *discounted return* is dominated by the rewards from the  near future
- $\gamma$ is close to 1 $\rightarrow$  dominated by the rewards from the far future

An *episode* is usually assumed to be *finite* *trajectory* (i.e. the agent stops at some *terminal states*). Task with episodes are called *episodic tasks*.



A positive reward represents encouragement to take such actions, a negative reward represents punishment to take such actions.



Converting episodic tasks to continuing tasks:

- Option 1: Treat the target state as a special *absorbing state*. Once the agent reaches an absorbing state, it will never leave and the consequent rewards $r = 0$.
- **Option 2:** Treat the target state as a normal state with a policy. The agent can still leave the target state and gain $r = +1$ when entering the target afterwards.



### MDP: Markov Decision Process

Sets:

- State: the set of states $S$.

-  Action: the set of actions $A(s)$ is associated for state $s \in S$.

- Reward: the set of rewards $R(s, a)$.

Probability distribution (or called *system model*):

- State transition probability: at state $s$, taking action $a$, the probability to
    transit to state $s'$ is $p(s' |s, a)$.
- Reward probability: at state $s$, taking action $a$, the probability to get
    reward $r$ is $p(r|s, a)$.

Policy: at state $s$, the probability to choose action $a$ is $\pi (a|s)$.

Markov property: memoryless property



### Bellman Equation

#### State Value

> The value of one state relies on the values of other states ——*Bootstrapping*

$$
\displaylines{
{\bf{trajectory : \ }} S_t \xrightarrow{A_t} R_{t+1}, S_{t+1} \xrightarrow{A_{t+1}} R_{t+2}, S_{t+2} \xrightarrow{A_{t+2}} R_{t+3}, S_{t+3}, \xrightarrow{A_{t+3}} \cdots \\
\begin{align}
{\bf{discounted \ return: \ }}G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots  = R_{t+1} + \gamma G_{t+1} \\
{\bf{state \ value: \ }}v_{\pi}(s) &= \Bbb{E}[G_t | S_t=s] \\
&= \Bbb{E}[R_{t+1} + \gamma G_{t+1} | S_t=s] \\
&= \Bbb{E}[R_{t+1} | S_t=s] + \gamma\Bbb{E}[G_{t+1} | S_t=s]
\end{align}
}
$$

First term of RHS (i.e. expectation of *immediate rewards*):
$$
\begin{align}
\Bbb{E}[R_{t+1} | S_t=s] &= \sum_{a}\pi(a|s)\Bbb{E}[R_{t+1}|S_t=a,A_t=a] \\
&=\sum_{a}\pi(a|s)\sum_{r}p(r|s,a)r \\
&\overset{\Delta}{=}r_{\pi}(s) \tag{1}
\end{align}
$$
Second term of RHS (i.e. expectation of *future rewards*):
$$
\begin{align}
\Bbb{E}[G_{t+1} | S_t=s] &= \sum_{s'}\Bbb{E}[G_{t+1}|S_{t+1}=s']p(s'|s) \\
&=\sum_{s'}v_{\pi}(s')p(s'|s) \\
&=\sum_{s'}v_{\pi}(s')\sum_{a}p(s'|s,a)\pi(a|s) \overset{\Delta}{=}\sum_{s'}v_{\pi}(s')p_{\pi}(s'|s) \tag{2}\\
&=\sum_{a}\pi(a|s)\sum_{s'}p(s'|s,a)v_{\pi}(s') \\
\end{align}
$$
#### Bellman Equation

Therefore the $\bf{Bellman \ Equation}$ can conclude from (1) and (2):
$$
\begin{align}
v_{\pi}(s) &=\sum_{a}\pi(a|s)\left[\sum_{r}p(r|s,a)r+\gamma \sum_{s'}v_{\pi}(s')p(s'|s,a)\right] \tag{3} \\
&= r_{\pi}(s) +\gamma \sum_{s'}v_{\pi}(s')p_{\pi}(s'|s) \\
&= r _ { \pi } \left( s _ { i } \right) + \gamma \sum _ { s _ { j } } p _ { \pi } \left( s _ { j }|s _ { i } \right) v _ { \pi } \left( s _ { j } \right) ;   \\
& \text{Suppose the states could be indexed as}\  s_i \ (i = 1, . . . , n).
\end{align}
$$
Matrix-vector form of $\bf{Bellman \ Equation}$:
$$
v _ { \pi } = r _ { \pi } + \gamma P _ { \pi } v _ { \pi }
$$

where

- $$v _ { \pi } = \left[ v _ { \pi } \left( s _ { 1 } \right) , \ldots , v _ { \pi } \left( s _ { n } \right) \right] ^ { T } \in \mathbb { R } ^ { n }$$ 
- $$r _ { \pi } = \left[ r _ { \pi } \left( s _ { 1 } \right) , \ldots , r _ { \pi } \left( s _ { n } \right) \right] ^ { T } \in \mathbb { R } ^ { n }$$ 
- $P _ { \pi } \in \mathbb { R } ^ { n \times n }$, where $\left[ P _ { \pi } \right] _ { i j } = p _ { \pi } \left( s _ { j } \mid s _ { i } \right)$ is the *state transition matrix*

example: 
$$
\underbrace{
  \begin{bmatrix}
  v_\pi(s_1) \\
  v_\pi(s_2) \\
  v_\pi(s_3) \\
  v_\pi(s_4)
  \end{bmatrix}
}_{v_\pi}
=
\underbrace{
  \begin{bmatrix}
  r_\pi(s_1) \\
  r_\pi(s_2) \\
  r_\pi(s_3) \\
  r_\pi(s_4)
  \end{bmatrix}
}_{r_\pi}
+ \gamma
\underbrace{
  \begin{bmatrix}
  p_\pi(s_1|s_1) & p_\pi(s_2|s_1) & p_\pi(s_3|s_1) & p_\pi(s_4|s_1) \\
  p_\pi(s_1|s_2) & p_\pi(s_2|s_2) & p_\pi(s_3|s_2) & p_\pi(s_4|s_2) \\
  p_\pi(s_1|s_3) & p_\pi(s_2|s_3) & p_\pi(s_3|s_3) & p_\pi(s_4|s_3) \\
  p_\pi(s_1|s_4) & p_\pi(s_2|s_4) & p_\pi(s_3|s_4) & p_\pi(s_4|s_4)
  \end{bmatrix}
}_{P_\pi}
\underbrace{
  \begin{bmatrix}
  v_\pi(s_1) \\
  v_\pi(s_2) \\
  v_\pi(s_3) \\
  v_\pi(s_4)
  \end{bmatrix}
}_{v_\pi}
$$


Solve the value via iterative solution:
$$
v _ { k+1 } = r _ { \pi } + \gamma P _ { \pi } v _ { k } \\
v_k \rightarrow v_{\pi},\quad k \rightarrow \infty
$$

#### Action Value

$$
q _ { \pi } ( s , a ) = \mathbb { E } \left[ G _ { t } | S _ { t } = s , A _ { t } = a \right]
$$

Conclude from (3) that:
$$
\begin{align}
v_{\pi}(s) &= \Bbb{E}[G_t | S_t=s] \\
&=\sum_{a}\mathbb { E } \left[ G _ { t } | S _ { t } = s , A _ { t } = a \right]\pi(a|s) \\
&=\sum_{a}\pi(a|s) \underbrace{ \left[\sum_{r}p(r|s,a)r+\gamma \sum_{s'}v_{\pi}(s')p(s'|s,a)\right] }_{q_\pi(s,a)} \\
\end{align}
$$
Hence,
$$
\begin{align}
v_{\pi}(s) &= \sum_{a}\pi(a|s) q_\pi(s,a) \tag{4} \\
q_\pi(s,a) &= \sum_{r}p(r|s,a)r+\gamma \sum_{s'}v_{\pi}(s')p(s'|s,a) \tag{5}
\end{align}
$$
(4) and (5) are the two sides of the same coin:

- (4) shows how to obtain *state values* from *action values*.
- (5) shows how to obtain *action values* from *state values* .



### BOE: Bellman Optimality Equation

#### Intro

Evaluating whether a policy is good: if
$$
v_{\pi 1}(s) \geqslant v_{\pi 2}(s) \quad \text{for all } s \in \cal{S}
$$
then $\pi_1$ is better than $\pi_2$.

Therefore a policy $\pi ^*$ is optimal if $v_{\pi ^*}(s) \geqslant v_{\pi}(s)$ for all $s$ and for any other policy $\pi$.



$\bf{Bellman\ Optimality \ Equation}$ (elementwise form):
$$
\begin{align}
v(s) &=\max _ { \pi } 
\sum_{a}\pi(a|s)\left[\sum_{r}p(r|s,a)r+\gamma \sum_{s'}v(s')p(s'|s,a)\right], \quad s \in \cal{S} \\
&= \max _ { \pi }\sum_{a}\pi(a|s) q(s,a), \quad s \in \cal{S}
\end{align}
$$
**This equation shows how the state values (LHS) achieves its maximum (RHS) which corresponds to the optimal policy.**

Remarks:

- $p(r|s,a)$, $p(s'|s,a)$, $r$, $\gamma$ are known $\rightarrow$ $q(s,a)$ depends on $v(s')$.
- $v(s)$, $v(s')$ are unknown and to be calculated.
- The goal is to solve the equation to find the optimal policy $\pi ^*$.



$\bf{Bellman\ Optimality \ Equation}$ (matrix-vector form):
$$
v =\max _ { \pi }( r _ { \pi } + \gamma P _ { \pi } v)
$$
#### How to Solve

- Maximize the RHS expression elementwisely：
  $$
  \max _ { \pi }\sum_{a}\pi(a|s) q(s,a) \\
  \leqslant \max_{a \in \cal{A} (s)}q(s,a) = q ( s , a ^* )
  $$
  where the optimality is achieved when
  $$
  \pi ( a | s ) = \left\{ \begin{array} { c c } 1 & a = a ^ { * } \\ 0 & a \neq a ^ { * } \end{array} \right. \quad where  \ a ^ { * } = \arg \max _ { a } q ( s , a )
  $$
  This is because the RHS expression is actually the weighted sum of all possible $q ( s , a )$ given a specific $s$. Since the weights (i.e. the possibility $\pi ( a | s )$ for all $a$) adds up to 1, we just need to put all the weights on the largest $q ( s , a )$ (i.e. assign the probability of 1 to $q ( s , a ^ * )$, and 0 to the rest $q ( s , a )$) to achieve the maximum.

- As for the matrix-vector form, rewrite BOE as: $v =\max _ { \pi }( r _ { \pi } + \gamma P _ { \pi } v) \overset{\Delta}{=} f(v)$

- According to *Contraction Mapping Theorem*，BOE is a *contraction mapping* with a unique  *fixed point* $v^*$. 

  According  to *value iteration algorithm*, the solution of BOE could be solved iteratively by 
  $$
  v_{k+1} = f(v_k) = \max _ { \pi }( r _ { \pi } + \gamma P _ { \pi } v_k)
  $$
  This sequence ${v_k }$ converges to $v ^∗$ exponentially fast given any initial guess $v_0$ . The convergence rate is determined by $\gamma$.

- Suppose 

  $$
  \pi ^{ \ast } = \arg \max \limits_{\pi }\left(r_{\pi } + \gamma P_{\pi }v^{ \ast }\right. )
  $$

  Then 
  $$
  v ^{ \ast } = r_{\pi^* } + \gamma P_{\pi^* }v^{ \ast }
  $$
  Therefore, $π ^∗$ is a policy and $v ^∗ = v_{π^∗}$ is the corresponding state value.

- It can be proved in math that for the *fixed point* $v^*$,
  $$
  v^* \geqslant v_\pi, \quad \forall \pi
  $$
  therefore $\pi ^*$ is the optimal policy. It is a *deterministic greedy policy* as below:
  $$
  \pi ( a | s ) = \left\{ \begin{array} { c c } 1 & a = a ^ { * } \\ 0 & a \neq a ^ { * } \end{array} \right.
  $$
  where $\ a ^ { * } = \arg \max _ { a } q^* ( s , a )$ and $q^* ( s , a )$ corresponds to $v^*(a)$.

> The optimal policies are invariant to the linear transformation of the reward signals.

### Truncated Policy Iteration Algorithm

*Pseudocode:*

<b>Initialization:</b> The probability model $p(r|s,a)$ and $p(s'|s,a)$ for all $(s,a)$ are known.

<b>Aim:</b> Search for the optimal state value and an optimal policy.

---

<b>While</b> $v_k$ has not converged, <b>for</b> the $k$th iteration $(k = 0, 1, 2, . . . )$, <b>do</b>

​	*Policy evaluation:*

​	<b>Initialization:</b> select the initial guess as $v_k^{(0)}=v_{k-1}$. The maximum iteration is set to be $j_{truncate}$.

​	<b>While</b> $j < j_{truncate}$, <b>do</b>

​		<b>For</b> every state $s \in \cal{S}$, <b>do</b>

​			$v_{k}^{(j + 1)}(s) = \sum _{a}\pi _{k}(a | s)\left[\sum _{r}p(r | s,a)⁢⁢r + \gamma \sum _{s^{′}}p(s^{′} | s,a)v_{k}^{(j)}(s^{′})\right]$

​	<b>Set</b> $v_k = v_k ^{(j_{truncate})}$		*# Note that when $j \rightarrow \infty$, $v_k ^{(j)}$ converges to $v_{\pi_k}$ of current $\pi_k $*

​							    *# and it can be proved in math that for any iteration $j$，$v_k^{j-1}<v_k^j<v_k^{j_{truncate}}$.*

​							    *# S.t. $v_{\pi_k}$ is taken as the evaluation of  $\pi_k $, and we improve $\pi_k $ based on $v_{\pi_k}$.*

​							    *# For computational efficiency, only iterate over $j < j_{truncate}$*

​							    *# because $v_k ^{(j_{truncate})}$ is close enough to $v_{\pi_k}$.* 

​	*Policy improvement:*

​	<b>For</b> every state $s \in \cal{S}$, <b>do</b>

​		<b>For</b> every action $a \in \cal{A}(s)$, <b>do</b>

​			$q_{k}(s,a) = \sum _{r}p(r | s,a)⁢⁢r + \gamma \sum _{s^{′}}p(s^{′} | s,a)v_{k}(s^{′})$

​  	 	$a_k^*(s) = \arg \max \limits_{a} q_k(s,a)$

​		$\pi_{k+1}(a|s) = 1$ if $a=a_k^*$, and $\pi_{k+1}(a|s) = 0$ otherwise

​							    *# Lemma: If $\pi_{k+1}=\arg \max \limits_{\pi }\left(r_{\pi } + \gamma P_{\pi }v_{\pi_k}\right. )$ then $v_{\pi_{k} }<v_{\pi_{k+1} } $ for any $k$.* 

​							    *# Theorem: The state value generated by the iteration converges to the optimal state value $v^{\ast}$,*

​							    *# as a result, the policy converges to an optimal policy.* 

---

The case of $j_{truncate} = 1$ is *Value Iteration Algorithm*, and the case of $j_{truncate} = \infty$ is *Policy Iteration Algorithm*.

<img src="fig1.png" style="zoom: 22%;" />

### MC: Monte Carlo Learning

> model-free RL: When model is unavailable, we can use data.

#### MC Basic algorithm

> Many model-based and model-free RL algorithms fall into this framework.

*Pseudocode:*

<b>Initialization: </b>Initial guess $\pi_0$.
<b>Aim:</b> Search for an optimal policy.

---

<b>For</b> the $k$th iteration $(k = 0, 1, 2, . . . )$, <b>do</b>

​	<b>For</b> every state $s \in \cal{S}$, <b>do</b>

​		Collect sufficiently many episodes starting from $(s,a)$ following $\pi_k $

​		*Policy evaluation:*

​		$q_{\pi_k}(s, a) \approx q_k(s,a)=$ average return of all the episodes starting from $(s,a)$

​	*Policy improvement:*

​	$a_k^*(s) = \arg \max \limits_{a} q_k(s,a)$

​	$\pi_{k+1}(a|s) = 1$ if $a=a_k^*$, and $\pi_{k+1}(a|s) = 0$ otherwise

---

Since policy iteration is convergent, the convergence of MC Basic is also guaranteed to be convergent given sufficient episodes.

Episode length:

- When the episode length is short, only the states that are close to the target have nonzero state values.
- As the episode length increases, the states that are closer to the target have nonzero values earlier than those farther away.
- The episode length should be sufficiently long, but does not have to be infinitely long.

<b>Visit:</b> every time a state-action pair appears in the episode, it is called a visit of that state-action pair.

For an episode such as:
$$
s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \dots
$$
Methods to approximate $q_{\pi_k}(s, a)$ using the data: 

- <b>Initial-visit method:</b> Just calculate the return and approximate $q_\pi(s_1, a_2)$.

- <b>Every-visit method:</b> 
  $$
  \begin{align*}
  s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \dots & \quad \text{[original episode]} \\
  \phantom{s_1 \xrightarrow{a_2}} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \dots & \quad \text{[episode starting from } (s_2, a_4)\text{]} \\
  \phantom{s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4}} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \dots & \quad \text{[episode starting from } (s_1, a_2)\text{]} \\
  \phantom{s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2}} s_2 \xrightarrow{a_3} s_5 \xrightarrow{a_1} \dots & \quad \text{[episode starting from } (s_2, a_3)\text{]} \\
  \phantom{s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3}} s_5 \xrightarrow{a_1} \dots & \quad \text{[episode starting from } (s_5, a_1)\text{]}
  \end{align*}
  $$

  Can estimate $q _ { \pi } \left( s _ { 1 } , a _ { 2 } \right) , q _ { \pi } \left( s _ { 2 } , a _ { 4 } \right) , q _ { \pi } \left( s _ { 2 } , a _ { 3 } \right) , q _ { \pi } \left( s _ { 5 } , a _ { 1 } \right) , \ldots$

When to update the policy:

- Low efficiency: In the policy evaluation step, to collect all the episodes starting from a state-action pair and then use the average return to approximate the action value.
- High efficiency: Uses the return of a single episode to approximate the action value. In this way, we can improve the policy episode-by-episode.
- Will the second method cause problems? 
  - In fact, we have applied its idea in the truncated policy iteration algorithm (i.e. approximating $v_{\pi_k}$ via limited iterations). 
  - Even if using the single episode we sampled as $q(s,a)$ is imprecise,  as long as the relative values of $q(s,a)$ for different $a$ are roughly correct, the policy still selects the action with higher return in $\arg \max \limits_{a} q_k(s,a)$, thus improving on the right track.
  - When the policy gets better, **it generates episodes that are closer to the optimal trajectory, and consequently less random (i.e. the variance of $q_k(s,a)$ gets smaller and thus $q_k(s,a)$ converges to $q_{\pi_k}(s, a)$ continuously)**.

#### MC Exploring Starts

*Pseudocode:*

<b>Initialization: </b> Initial policy $\pi_0(a|s)$ and initial value $q(s,a)$ for all $(s,a)$. $Returns(s,a) = [ \ ]$ for all $(s,a)$.

<b>Aim:</b> Search for an optimal policy.

---

<b>For</b> each episode, <b>do</b>

​	*Episode generation:* Select a starting state-action pair $(s_0, a_0)$ and ensure that all pairs can be possibly selected (this is the exploring-starts condition). Following the current policy, generate an episode of length $T$: $s_0, a_0,r_1, ..., s_{T-1}, a_{T-1},r_{T}$.

​	<b>Initialization</b> for each episode: $g \leftarrow 0$

<b>	For</b> each step of the episode, $t = T-1, T-2, ... , 0$, <b>do</b>	*# Compute reversely from the end of the episode.*

​		$g \leftarrow \gamma g + r_{t+1}$							         		*#  S.t. only need one step of calculation for each update of $g$.*

​		$Returns(s_t, a_t) \leftarrow Returns(s_t,a_t) \cup \{g\}$

​		*Policy evaluation:*

​		$q(s_t,a_t) \leftarrow$ average($Returns(s_t,a_t)$)

​		*Policy improvement:*

​		$\pi(a|s_t) = 1$ if $a=\arg \max \limits_{a} q(s_t, a)$, and $\pi(a|s_t) = 0$ otherwise

---

> What is exploring starts? Exploring starts means we need to generate sufficiently many episodes $\underbrace{starting}_{starts}$ from $\underbrace{every}_{exploring}$ state-action pair.

In theory, only if every action value for every state is well explored, can we select the optimal actions correctly. Otherwise, if an action is not explored, this action may happen to be the optimal one and hence be missed.

#### MC $\varepsilon$-Greedy

> What is a soft policy? A policy is *soft* if the probability to take any action is positive. With a soft policy, a few episodes that are sufficiently long can visit every state-action pair. 
>
> Then, we do not need to have a large number of episodes starting from every state-action pair. Hence, the requirement of exploring starts can be removed.
>
> - Deterministic policy: e.g. greedy policy
> - Stochastic policy: e.g. soft policy

*Pseudocode:*

<b>Initialization: </b> Initial policy $\pi_0(a|s)$ and initial value $q(s,a)$ for all $(s,a)$. $Returns(s,a) = [ \ ]$ for all $(s,a)$. $\varepsilon \in (0,1]$.

<b>Aim:</b> Search for the optimal state value and an optimal policy.

---

<b>For</b> each episode, <b>do</b>

​	*Episode generation:* Select a starting state-action pair $(s_0, a_0)$ (the exploring-starts condition is not required). Following the current policy, generate an episode of length $T$: $s_0, a_0,r_1, ..., s_{T-1}, a_{T-1},r_{T}$.

​	<b>Initialization</b> for each episode: $g \leftarrow 0$

​	<b>	For</b> each step of the episode, $t = T-1, T-2, ... , 0$, <b>do</b>

​		$g \leftarrow \gamma g + r_{t+1}$

​		$Returns(s_t, a_t) \leftarrow Returns(s_t,a_t) \cup \{g\}$

​		*Policy evaluation:*

​		$q(s_t,a_t) \leftarrow$ average($Returns(s_t,a_t)$)

​		*Policy improvement:*		

​		Let $a^*=\arg \max \limits_{a} q(s_t, a)$ and 
$$
\pi ( a | s _ { t } ) = \left\{ \begin{array} { c c } 1 - \frac { \left| \mathcal { A } \left( s _ { t } \right) \right| - 1 } { \left| \mathcal { A } \left( s _ { t } \right) \right| } \epsilon , & a = a ^ { * } \\ \frac { 1 } { \left| \mathcal { A } \left( s _ { t } \right) \right| } \epsilon , & a \neq a ^ { * } \end{array} \right.
$$

---

$\varepsilon$ controls the balance between <b>exploration</b> and <b>exploitation</b>.

The advantage of $ε$-greedy policies is that they have strong exploration ability when $ε$ is large.

The disadvantage is that $ε$-greedy polices are not optimal in general.

- $ε$ cannot be too large. We can also use a decaying $ε$.



### SA: Stochastic Approximation

SA refers to a broad class of stochastic iterative algorithms solving root finding or optimization problems.

Compared to many other root-finding algorithms such as gradient-based methods, SA is powerful in the sense that it <b>does not require to know the expression of the objective function nor its derivative</b> (model-free).

#### Robbins-Monro algorithm

<b>Problem statement:</b> Suppose we would like to find the root of the equation
$$
g(w)=0
$$
The <b>Robbins-Monro (RM) algorithm</b> that can solve this problem is as follows:
$$
w _ { k + 1 } = w _ { k } - a _ { k } \tilde { g } ( w _ { k } , \eta _ { k } ) , \quad k = 1 , 2 , 3 , \ldots
$$
where

- $w_k$is the kth estimate of the root

- $\tilde { g } ( w _ { k } , \eta _ { k } ) = g(w_k)+\eta_k$ is the $k$th noisy observation

- $a_k$ is a positive coefficient.

The function $g(w)$ is viewed as a black box, for which only the input sequence $\{w_k\}$ and output sequence (noisy) $\{\tilde { g } ( w _ { k } , \eta _ { k } ) \}$ are available. So this algorithm relies on data instead of model.



<b>Robbins-Monro Theorem</b>

---

In the Robbins-Monro algorithm, if

1.  $0 < c _ { 1 } \leq \nabla _ { w } g ( w ) \leq c _ { 2 }$ for all $w $; 
2.  $\sum _ { k = 1 } ^ { \infty } a _ { k } = \infty$ and $\sum _ { k = 1 } ^ { \infty } a _ { k } ^ { 2 } < \infty $; 
3.  $\mathbb { E } \left[ \eta _ { k } \mid \mathcal { H } _ { k } \right] = 0$ and $\mathbb { E } \left[ \eta _ { k } ^ { 2 } \mid \mathcal { H } _ { k } \right] < \infty ;$ 

where $$\mathcal { H } _ { k } = \left\{ w _ { k } , w _ { k - 1 } , \ldots \right\}$$, then $w_k$  converges w.p.1 to the root $w^*$ satisfying $g(w^*) = 0$.

---

Explanations of the three conditions:

- Condition1：
  - $g$ should be monotonically increasing, which ensures that the root of $g(w) = 0$ exists and is unique.
  - This condition requires that $g(w)$ is convex.

- Condition 2:

  - $\sum _ { k = 1 } ^ { \infty } a _ { k }^2 < \infty$ ensures that $a_k$ converges to zero as $k \rightarrow \infty$, so $w_k$ converges to $w^*$  as well. 

    Also, if $w _ { k } \rightarrow w ^ { * } , g ( w _ { k } ) \rightarrow 0$ and $\tilde { g } ( w _ { k } , \eta _ { k } )$ is dominant by $\eta _ { k } $. This randomness should be limited.

  - $\sum _ { k = 1 } ^ { \infty } a _ { k } = \infty$ ensures that $a_k$ do not converge to zero too fast. Otherwise  $a_k$ might converge to zero too early when there is still quite a distance between $w_k$ and $w^*$, in this case $w_k$ is not able to get closer to $w^*$.

- Condition 3: The noise should be unbiased and its variance should be limited.

#### BGD, SGD, MBGD

<b>Problem setup:</b> Suppose we aim to solve the following optimization problem:
$$
\min _ { w }  J ( w ) = \mathbb { E } [ f ( w , X ) ]
$$

- $w$ is the parameter to be optimized.
- $X$ is a random variable. The expectation is with respect to $X$.
- $w$ and $X$ can be either scalars or vectors. The function $f (·)$ is a scalar.

The *Batch Gradient Descent*, *Stochastic Gradient Descent*, *Mini-batch Gradient Descent* algorithms solving this problem are, respectively,
$$
w _ { k + 1 } = w _ { k } - \alpha _ { k } \underbrace{\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \nabla _ { w } f ( w _ { k } , x _ { i } )}_{\approx  \mathbb { E } \left[ \nabla _ { w } f \left( w _ { k } , X \right) \right] } , \quad\quad (BGD) \\
$$
All the samples are used in every iteration, so the approximation to the true gradient $\mathbb { E } \left[ \nabla _ { w } f \left( w _ { k } , X \right) \right]$ is close.
$$
w _ { k + 1 } = w _ { k } - \alpha _ { k } \underbrace{\frac { 1 } { m } \sum _ { j \in \mathcal { I } _ { k } } \nabla _ { w } f ( w _ { k } , x _ { j } ) }_{\approx  \mathbb { E } \left[ \nabla _ { w } f \left( w _ { k } , X \right) \right] }, \quad\quad(MBGD) \\
$$
$\cal{I}_k$ is a subset of $\{1, . . . , n\}$ with the size as $|\cal{I}_k | = m$. The set  $\cal{I}_k$ is obtained by $m$ times idd samplings.
$$
w _ { k + 1 } = w _ { k } - \alpha _ { k } \nabla _ { w } f ( w _ { k } , x _ { k } ). \quad\quad (SGD)
$$

<img src="fig2.png" style="zoom: 40%;" />



### TD: Temporal Difference Learning

Problem statement:

- Given policy $\pi$, the aim is to estimate the state values $\{v_\pi (s)\}_{s\in S}$ <b>under $\pi$ </b>.

<b>The TD learning algorithm is</b>
$$
\begin{align}
\underbrace{v_{t + 1}(s_{t})}_{new \ estimate} &= \underbrace{v_{t}(s_{t})}_{current \ estimate}−\alpha _{t}(s_{t})\underbrace{[v_{t}(s_{t})−[\overbrace{ r_{t + 1} + \gamma v_{t}(s_{t + 1})}^{TD \ target \ \bar{v}_t}]}_{TD \ error \ \delta_t } \tag{1}\\
v _ { t + 1 } ( s ) &= v _ { t } ( s ) , \quad \forall s \neq s _ { t } , \tag{2}
\end{align}
$$
where $t = 0,1,2,...$

- Here, $v_t (s_t )$ is the estimated state value of $v_\pi (s_t )$; $\alpha_t (s_t )$ is the learning rate of $s_t$ at time $t$.
- At time $t$, only the value of the visited state $s_t$ is updated whereas the values of the unvisited states $s \neq  s_t$ remain unchanged.
- The update in $(2)$ will be omitted when the context is clear.

Observation: The new estimate $v_{t + 1}(s_{t})$ is a combination of the current estimate $v_{t}(s_{t})$ and the TD error.

- Interpretation of *TD target $\bar{v}_t$*: $v_{t}(s_{t})$ is the estimate for state value at $s_t$ before taking the action $a_t$. At $s_{t+1}$ (i.e. after taking $a_t$), the agent gets the true reward $r_{t+1}$ along with the estimate $v_{t}(s_{t+1})$ for the new state $s_{t+1}$. Then $r_{t + 1} + \gamma v_{t}(s_{t + 1})$ is considered more precise in estimating the true state value $v_\pi(s_t)$ (<b>as it contains the real feedback of a step forward</b>), and thus becomes the *target* for updating $v_{t}(s_{t})$.

- Concluded from above, TD error can be interpreted as <b>innovation</b>, which means new information obtained from the experience.

- At every time step, the current estimate $v_{t}(s_{t})$ is updated by subtracting the error $\delta_t$ to TD target $\bar{v}_t$, therefore $v_{t}(s_{t})$ is driven towards $\bar{v}_t$.

- If $v_t$ = $v_\pi$ , then $\delta_t$ should be zero (in the expectation sense), i.e.
  $$
  \mathbb { E } \left[ \delta _ { \pi , t } | S _ { t } = s _ { t } \right] = v _ { \pi } ( s _ { t } ) - \mathbb { E } \left[ R _ { t + 1 } + \gamma v _ { \pi } ( S _ { t + 1 } ) | S _ { t } = s _ { t } \right] = 0
  $$
  Hence, if $\delta_t$ is not zero, then $v_t$ is not equal to $v_\pi$ .

<b>Other properties</b>: The TD algorithm in $(1)$ <b>only estimates the state value</b> of a given policy. It does not estimate the action values nor search for optimal policies.



