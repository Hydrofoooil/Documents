# Math Foundation RL

## <span style="color:#e5b567;">Basic Concepts</span>

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



## <span style="color:#e5b567;">Bellman Equation</span>

### State Value

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
### Bellman Equation

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

$$v _ { \pi } = \left[ v _ { \pi } \left( s _ { 1 } \right) , \ldots , v _ { \pi } \left( s _ { n } \right) \right] ^ { T } \in \mathbb { R } ^ { n }$$ 
$$r _ { \pi } = \left[ r _ { \pi } \left( s _ { 1 } \right) , \ldots , r _ { \pi } \left( s _ { n } \right) \right] ^ { T } \in \mathbb { R } ^ { n }$$
    $P _ { \pi } \in \mathbb { R } ^ { n \times n }$, where $\left[ P _ { \pi } \right] _ { i j } = p _ { \pi } \left( s _ { j } \mid s _ { i } \right)$ is the *state transition matrix*

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

### Action Value

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



## <span style="color:#e5b567;">BOE: Bellman Optimality Equation</span>

### Intro

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
### How to Solve

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

- It can be proved mathematically that for the *fixed point* $v^*$,
  $$
  v^* \geqslant v_\pi, \quad \forall \pi
  $$
  therefore $\pi ^*$ is the optimal policy. It is a *deterministic greedy policy* as below:
  $$
  \pi ( a | s ) = \left\{ \begin{array} { c c } 1 & a = a ^ { * } \\ 0 & a \neq a ^ { * } \end{array} \right.
  $$
  where $\ a ^ { * } = \arg \max _ { a } q^* ( s , a )$ and $q^* ( s , a )$ corresponds to $v^*(a)$.

> The optimal policies are invariant to the linear transformation of the reward signals.

## <span style="color:#e5b567;">Truncated Policy Iteration Algorithm</span>

*Pseudocode:*

<b>Initialization:</b> The probability model $p(r|s,a)$ and $p(s'|s,a)$ for all $(s,a)$ are known.

<b>Aim:</b> Search for the optimal state value and an optimal policy.

---

<b>While</b> $v_k$ has not converged, <b>for</b> the $k$th iteration $(k = 0, 1, 2, . . . )$, <b>do</b>

​&emsp;&emsp;*Policy evaluation:*

&emsp;&emsp;<b>Initialization:</b> select the initial guess as $v_k^{(0)}=v_{k-1}$. The maximum iteration is set to be $j_{truncate}$.

​&emsp;&emsp;<b>While</b> $j < j_{truncate}$, <b>do</b>

&emsp;&emsp;&emsp;&emsp;<b>For</b> every state $s \in \cal{S}$, <b>do</b>

​&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$v_{k}^{(j + 1)}(s) = \sum _{a}\pi _{k}(a | s)\left[\sum _{r}p(r | s,a)⁢⁢r + \gamma \sum _{s^{′}}p(s^{′} | s,a)v_{k}^{(j)}(s^{′})\right]$

&emsp;&emsp;<b>Set</b> $v_k = v_k ^{(j_{truncate})}$  *# Note that when $j \rightarrow \infty$, $v_k ^{(j)}$ converges to $v_{\pi_k}$ of current $\pi_k$*

​							        *# and it can be proved mathematically that for any iteration $j$，$v_k^{j-1}<v_k^j<v_k^{j_{truncate}}$.*

​							        *# S.t. $v_{\pi_k}$ is taken as the evaluation of  $\pi_k$, and we improve $\pi_k$ based on $v_{\pi_k}$.*

​							        *# For computational efficiency, only iterate over $j < j_{truncate}$*

​							        *# because $v_k ^{(j_{truncate})}$ is close enough to $v_{\pi_k}$.* 

​&emsp;&emsp;*Policy improvement:*

&emsp;&emsp;<b>For</b> every state $s \in \cal{S}$, <b>do</b>

&emsp;&emsp;&emsp;&emsp;<b>For</b> every action $a \in \cal{A}(s)$, <b>do</b>

​&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$q_{k}(s,a) = \sum _{r}p(r | s,a)⁢⁢r + \gamma \sum _{s^{′}}p(s^{′} | s,a)v_{k}(s^{′})$

​&emsp;&emsp;&emsp;&emsp;$a_k^*(s) = \arg \max \limits_{a} q_k(s,a)$

​&emsp;&emsp;&emsp;&emsp;$\pi_{k+1}(a|s) = 1$ if $a=a_k^*$, and $\pi_{k+1}(a|s) = 0$ otherwise

​							    *# Lemma: If $\pi_{k+1}=\arg \max \limits_{\pi }\left(r_{\pi } + \gamma P_{\pi }v_{\pi_k}\right. )$ then $v_{\pi_{k} }<v_{\pi_{k+1} }$ for any $k$.* 

​							    *# Theorem: The state value generated by the iteration converges to the optimal state value $v^{\ast}$,*

​							    *# as a result, the policy converges to an optimal policy.* 

---

The case of $j_{truncate} = 1$ is *Value Iteration Algorithm*, and the case of $j_{truncate} = \infty$ is *Policy Iteration Algorithm*.

<img src="fig1.png" style="zoom: 30%;" />

## <span style="color:#e5b567;">MC: Monte Carlo Learning</span>

> model-free RL: When model is unavailable, we can use data.

### MC Basic algorithm

> Many model-based and model-free RL algorithms fall into this framework.

*Pseudocode:*

<b>Initialization: </b>Initial guess $\pi_0$.
<b>Aim:</b> Search for an optimal policy.

---

<b>For</b> the $k$th iteration $(k = 0, 1, 2, . . . )$, <b>do</b>

​&emsp;&emsp;<b>For</b> every state $s \in \cal{S}$, <b>do</b>

​&emsp;&emsp;&emsp;&emsp;Collect sufficiently many episodes starting from $(s,a)$ following $\pi_k$

​&emsp;&emsp;&emsp;&emsp;*Policy evaluation:*

​&emsp;&emsp;&emsp;&emsp;$q_{\pi_k}(s, a) \approx q_k(s,a)=$ average return of all the episodes starting from $(s,a)$

​&emsp;&emsp;*Policy improvement:*

​&emsp;&emsp;$a_k^*(s) = \arg \max \limits_{a} q_k(s,a)$

​&emsp;&emsp;$\pi_{k+1}(a|s) = 1$ if $a=a_k^*$, and $\pi_{k+1}(a|s) = 0$ otherwise

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

### MC Exploring Starts

*Pseudocode:*

<b>Initialization: </b> Initial policy $\pi_0(a|s)$ and initial value $q(s,a)$ for all $(s,a)$. $Returns(s,a) = [ \ ]$ for all $(s,a)$.

<b>Aim:</b> Search for an optimal policy.

---

<b>For</b> each episode, <b>do</b>

​&emsp;&emsp;*Episode generation:* Select a starting state-action pair $(s_0, a_0)$ and ensure that all pairs can be possibly selected as starting point (this is the exploring-starts condition). Following the current policy, generate an episode of length $T$: $s_0, a_0,r_1, ..., s_{T-1}, a_{T-1},r_{T}$.

​&emsp;&emsp;<b>Initialization</b> for each episode: $g \leftarrow 0$

&emsp;&emsp;<b>For</b> each step of the episode, $t = T-1, T-2, ... , 0$, <b>do</b>	*# Compute reversely from the end of the episode.*

​&emsp;&emsp;&emsp;&emsp;$g \leftarrow \gamma g + r_{t+1}$							         	*#  S.t. only need one step of calculation for each update of $g$.*

​&emsp;&emsp;&emsp;&emsp;$Returns(s_t, a_t) \leftarrow Returns(s_t,a_t) \cup \{g\}$

​&emsp;&emsp;&emsp;&emsp;*Policy evaluation:*

​&emsp;&emsp;&emsp;&emsp;$q(s_t,a_t) \leftarrow$ average($Returns(s_t,a_t)$)

​&emsp;&emsp;&emsp;&emsp;*Policy improvement:*

​&emsp;&emsp;&emsp;&emsp;$\pi(a|s_t) = 1$ if $a=\arg \max \limits_{a} q(s_t, a)$, and $\pi(a|s_t) = 0$ otherwise

---

> What is exploring starts? Exploring starts means we need to generate sufficiently many episodes $\underbrace{starting}_{starts}$ from $\underbrace{every}_{exploring}$ state-action pair.

In theory, only if every action value for every state is well explored, can we select the optimal actions correctly. Otherwise, if an action is not explored, this action may happen to be the optimal one and hence be missed.

### MC $\varepsilon$-Greedy

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

​&emsp;&emsp;*Episode generation:* Select a starting state-action pair $(s_0, a_0)$ (the exploring-starts condition is not required). Following the current policy, generate an episode of length $T$: $s_0, a_0,r_1, ..., s_{T-1}, a_{T-1},r_{T}$.

​&emsp;&emsp;<b>Initialization</b> for each episode: $g \leftarrow 0$

​&emsp;&emsp;<b>For</b> each step of the episode, $t = T-1, T-2, ... , 0$, <b>do</b>

​&emsp;&emsp;&emsp;&emsp;$g \leftarrow \gamma g + r_{t+1}$

​&emsp;&emsp;&emsp;&emsp;$Returns(s_t, a_t) \leftarrow Returns(s_t,a_t) \cup \{g\}$

​&emsp;&emsp;&emsp;&emsp;*Policy evaluation:*

​&emsp;&emsp;&emsp;&emsp;$q(s_t,a_t) \leftarrow$ average($Returns(s_t,a_t)$)

&emsp;&emsp;&emsp;&emsp;*Policy improvement:*		

​&emsp;&emsp;&emsp;&emsp;Let $a^*=\arg \max \limits_{a} q(s_t, a)$ and 
$$
\pi ( a | s _ { t } ) = \left\{ \begin{array} { c c } 1 - \frac { \left| \mathcal { A } \left( s _ { t } \right) \right| - 1 } { \left| \mathcal { A } \left( s _ { t } \right) \right| } \epsilon , & \text{for the greedy action} \\ \frac { 1 } { \left| \mathcal { A } \left( s _ { t } \right) \right| } \epsilon , & \text{fot the other } |\cal{A}\left(s\right)|-1 \text{ actions} \end{array} \right.
$$

---

$\varepsilon$ controls the balance between <b>exploration</b> and <b>exploitation</b>.

The advantage of $ε$-greedy policies is that they have strong exploration ability when $ε$ is large.

The disadvantage is that $ε$-greedy polices are not optimal in general.

- $ε$ cannot be too large. We can also use a decaying $ε$.



## <span style="color:#e5b567;">SA: Stochastic Approximation</span>

SA refers to a broad class of stochastic iterative algorithms solving root finding or optimization problems.

Compared to many other root-finding algorithms such as gradient-based methods, SA is powerful in the sense that it <b>does not require to know the expression of the objective function nor its derivative</b> (model-free).

### Robbins-Monro algorithm

<b>Problem statement:</b> Suppose we would like to find the root of the equation
$$
g(w)=0
$$
The <b>Robbins-Monro (RM) algorithm</b> that can solve this problem is as follows:
$$
w _ { k + 1 } = w _ { k } - a _ { k } \tilde { g } ( w _ { k } , \eta _ { k } ) , \quad k = 1 , 2 , 3 , \ldots
$$
where

- $w_k$ is the $k$th estimate of the root

- $\tilde { g } ( w _ { k } , \eta _ { k } ) = g(w_k)+\eta_k$ is the $k$th noisy observation

- $a_k$ is a positive coefficient.

The function $g(w)$ is viewed as a black box, for which only the input sequence $\{w_k\}$ and output sequence (noisy) $\{\tilde { g } ( w _ { k } , \eta _ { k } ) \}$ are available. So this algorithm relies on data instead of model.



<b>Robbins-Monro Theorem</b>

---

In the Robbins-Monro algorithm, if

1.  $0 < c _ { 1 } \leq \nabla _ { w } g ( w ) \leq c _ { 2 }$ for all $w$; 
2.  $\sum _ { k = 1 } ^ { \infty } a _ { k } = \infty$ and $\sum _ { k = 1 } ^ { \infty } a _ { k } ^ { 2 } < \infty$; 
3.  $\mathbb { E } \left[ \eta _ { k } \mid \mathcal { H } _ { k } \right] = 0$ and $\mathbb { E } \left[ \eta _ { k } ^ { 2 } \mid \mathcal { H } _ { k } \right] < \infty ;$ 

where $$\mathcal { H } _ { k } = \left\{ w _ { k } , w _ { k - 1 } , \ldots \right\},$$then $w_k$  converges w.p.1 to the root $w^*$ satisfying $g(w^*) = 0$.

---

Explanations of the three conditions:

- Condition1：
  - $g$ should be monotonically increasing, which ensures that the root of $g(w) = 0$ exists and is unique.
  - This condition requires that $g(w)$ is convex.

- Condition 2:

  - $\sum _ { k = 1 } ^ { \infty } a _ { k }^2 < \infty$ ensures that $a_k$ converges to zero as $k \rightarrow \infty$, so $w_k$ converges to $w^*$  as well. 

    Also, if $w _ { k } \rightarrow w ^ { * } , g ( w _ { k } ) \rightarrow 0$ and $\tilde { g } ( w _ { k } , \eta _ { k } )$ is dominant by $\eta _ { k }$. This randomness should be limited.

  - $\sum _ { k = 1 } ^ { \infty } a _ { k } = \infty$ ensures that $a_k$ do not converge to zero too fast. Otherwise  $a_k$ might converge to zero too early when there is still quite a distance between $w_k$ and $w^*$, and in this case $w_k$ is not able to get closer to $w^*$.

- Condition 3: The noise should be unbiased and its variance should be limited.

### BGD, SGD, MBGD

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
$\cal{I}_k$ is a subset of $\{1, . . . , n\}$ with the size $|\cal{I}_k | = m$. The set  $\cal{I}_k$ is obtained by $m$ times idd samplings.
$$
w _ { k + 1 } = w _ { k } - \alpha _ { k } \nabla _ { w } f ( w _ { k } , x _ { k } ). \quad\quad (SGD)
$$

<img src="fig2.png" style="zoom: 40%;" />



## <span style="color:#e5b567;">TD: Temporal Difference Learning</span>

Problem statement:

- Given policy $\pi$, the aim is to estimate the state values $\{v_\pi (s)\}_{s\in S}$ <b>under $\pi$</b>.

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

- Interpretation of *TD target $\bar{v}_t$*: $v_{t}(s_{t})$ is the estimate for state value at $s_t$ before taking the action $a_t$. At $s_{t+1}$ (i.e. after taking $a_t$), the agent gets the true reward $r_{t+1}$ along with the estimate $v_{t}(s_{t+1})$ for the new state $s_{t+1}$. Then $r_{t + 1} + \gamma v_{t}(s_{t + 1})$ is considered more precise in estimating the true state value $v_\pi(s_t)$ (<b>as it contains the real feedback from a step forward</b>), and thus becomes the *target* for updating $v_{t}(s_{t})$.

- Concluded from above, TD error can be interpreted as <b>innovation</b>, which means new information obtained from the experience.

- At every time step, the current estimate $v_{t}(s_{t})$ is updated by subtracting the error $\delta_t$ to TD target $\bar{v}_t$, therefore $v_{t}(s_{t})$ is driven towards $\bar{v}_t$.

- If $v_t$ = $v_\pi$ , then $\delta_t$ should be zero (in the expectation sense), i.e.
  $$
  \mathbb { E } \left[ \delta _ { \pi , t } | S _ { t } = s _ { t } \right] = v _ { \pi } ( s _ { t } ) - \mathbb { E } \left[ R _ { t + 1 } + \gamma v _ { \pi } ( S _ { t + 1 } ) | S _ { t } = s _ { t } \right] = 0
  $$
  Hence, if $\delta_t$ is not zero, then $v_t$ is not equal to $v_\pi$ .

<b>Other properties</b>: The TD algorithm in $(1)$ <b>only estimates the state value</b> of a given policy. It does not estimate the action values nor search for optimal policies.

### Sarsa

Sarsa is the abbreviation of  **state-action-reward-state-action**.

The aim is to estimate the action values of a given policy $\pi$.

Suppose we have some experience $\left\{ \left( s _ { t } , a _ { t } , r _ { t + 1 } , s _ { t + 1 } , a _ { t + 1 } \right) \right\} _ { t }$ .

We can use the following Sarsa algorithm to estimate the action values:

$$
q_{t + 1}\left(s_{t},a_{t}\right) = q_{t}\left(s_{t},a_{t}\right)−\alpha _{t}\left(s_{t},a_{t}\right)\left[q_{t}\left(s_{t},a_{t}\right)−\left[r_{t + 1} + \gamma q_{t}\left(s_{t + 1},a_{t + 1}\right)\right]\right. 
$$
$$
q _ { t + 1 } ( s , a ) = q _ { t } ( s , a ) , \quad \forall ( s , a ) \neq \left( s _ { t } , a _ { t } \right) ,
$$
where $t=0,1,2,\dots$
- $q_{t}\left(s_{t},a_{t}\right)$ is an estimate of $q_{\pi}\left(s_{t},a_{t}\right)$；
- $\alpha _{t}\left(s_{t},a_{t}\right)$ is the learning rate depending on $s_t,a_t$.

> Note that the relationship between Sarsa and the previous TD learning  algorithm is that We can obtain Sarsa by replacing the state value estimate $v(s)$  in the TD algorithm with the action value estimate $q(s, a)$. As a result,  **Sarsa is an action-value version of the TD algorithm**.

The expression of  Sarsa suggests that it is a stochastic approximation algorithm solving the following equation:

$$q _ { \pi } ( s , a ) = \mathbb { E } \left[ R + \gamma q _ { \pi } \left( S ^ { \prime } , A ^ { \prime } \right) \mid s , a \right] , \quad \forall s , a .$$
This is another expression of the Bellman equation expressed in terms of action values.

Since the ultimate goal of RL is to find optimal policies, we combine Sarsa with a policy improvement step.  The combined algorithm is also called Sarsa.

*Pseudocode:*

---
<b>For</b> each episode, <b>do</b>

&emsp;&emsp;If the current $s_t$ is not the target state, do

&emsp;&emsp;&emsp;&emsp;Collect the experience $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$: In particular, take action $a_t$ following $\pi_t(s_t)$, generate $r_{t+1}$, $s_{t+1}$, and then take action $a_{t+1}$ following $\pi_t(s_{t+1})$.
&emsp;&emsp;&emsp;&emsp;*Update q-value:*
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) \Big[ q_t(s_t, a_t) - [r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})] \Big]$
&emsp;&emsp;&emsp;&emsp;*Update policy:*
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\pi_{t+1}(a|s_t) = 1 - \frac{\epsilon}{|\mathcal{A}|}(|\mathcal{A}| - 1) \text{ if } a = \arg\max_a q_{t+1}(s_t, a)$
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\pi_{t+1}(a|s_t) = \frac{\epsilon}{|\mathcal{A}|} \text{ otherwise}$

---

<img src="fig3.png" style="zoom: 40%;" />

### Expected Sarsa

A variant of Sarsa is the Expected Sarsa algorithm:

$$q _ { t + 1 } \left( s _ { t } , a _ { t } \right) = q _ { t } \left( s _ { t } , a _ { t } \right) - \alpha _ { t } \left( s _ { t } , a _ { t } \right) \left[ q _ { t } \left( s _ { t } , a _ { t } \right) - \left( r _ { t + 1 } + \gamma \mathbb { E } \left[ q _ { t } \left( s _ { t + 1 } , A \right) \right] \right) \right] ,$$
$$q _ { t + 1 } ( s , a ) = q _ { t } ( s , a ) , \quad \forall ( s , a ) \neq \left( s _ { t } , a _ { t } \right) ,$$ 
where$$𝔼\left[q_{t}\left(s_{t + 1},A\right)\right] = \sum _{a}\pi _{t}\left(a \mid s_{t + 1}\right)q_{t}\left(s_{t + 1},a\right) \doteq v_{t}\left(s_{t + 1}\right)$$ 
**Compared to Sarsa:**

- The *TD target* is changed into expectation form.
- Need more computation. But it is beneficial in the sense that it reduces the  estimation variances because it reduces random variables in Sarsa from $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ to $(s_t, a_t, r_{t+1}, s_{t+1})$.

### $n$-step Sarsa

$n$-step Sarsa: can unify Sarsa and Monte Carlo learning.

The definition of action value is

$$q_\pi(s, a) = \mathbb{E}[G_t|S_t = s, A_t = a].$$

The discounted return $G_t$ can be written in different forms as

$$\begin{array}{r c l} \text{Sarsa} & \longleftarrow & G_t^{(1)} = R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}), \\ & & G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 q_\pi(S_{t+2}, A_{t+2}), \\ & & \quad \vdots \\ n\text{-step Sarsa} & \longleftarrow & G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^n q_\pi(S_{t+n}, A_{t+n}), \\ & & \quad \vdots \\ \text{MC} & \longleftarrow & G_t^{(\infty)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \end{array}$$

It should be noted that $G_t = G_t^{(1)} = G_t^{(2)} = G_t^{(n)} = G_t^{(\infty)}$, where the superscripts merely indicate the different decomposition structures of $G_t$.

- Sarsa aims to solve
    
$$q_\pi(s, a) = \mathbb{E}[G_t^{(1)}|s, a] = \mathbb{E}[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1})|s, a].$$

- MC learning aims to solve
    
$$q_\pi(s, a) = \mathbb{E}[G_t^{(\infty)}|s, a] = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots |s, a].$$

- An intermediate algorithm called $n$-step Sarsa aims to solve
    
$$q_\pi(s, a) = \mathbb{E}[G_t^{(n)}|s, a] = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \dots + \gamma^n q_\pi(S_{t+n}, A_{t+n})|s, a].$$

- The algorithm of $n$-step Sarsa is    
$$\begin{aligned} q_{t+1}(s_t, a_t) = \ & q_t(s_t, a_t) - \alpha_t(s_t, a_t)\Big[q_t(s_t, a_t) - [r_{t+1} + \gamma r_{t+2} + \dots + \gamma^n q_t(s_{t+n}, a_{t+n})]\Big]. \end{aligned}$$

$n$-step Sarsa is more general because it becomes the (one-step) Sarsa algorithm when $n=1$ and the MC learning algorithm when $n=\infty$.
### Q-learning

Introduce **on-policy** learning and **off-policy** learning:

There exist two policies in a TD learning task:  
- The behavior policy is used to generate experience samples.  
- The target policy is constantly updated toward an optimal policy.  

On-policy vs off-policy:  
- When the behavior policy is the same as the target policy, such kind of learning is called on-policy.  
- When they are different, the learning is called off-policy.

Evaluating action value by Q-learning: 
$$q _ { t + 1 } \left( s _ { t } , a _ { t } \right) = q _ { t } \left( s _ { t } , a _ { t } \right) - \alpha _ { t } \left( s _ { t } , a _ { t } \right) \Big[ q _ { t } \left( s _ { t } , a _ { t } \right) - [ r _ { t + 1 } + \gamma \max _ { a \in \mathcal { A } } q _ { t } \left( s _ { t + 1 } , a \right) ]\Big] ,$$
$$q _ { t + 1 } ( s , a ) = q _ { t } ( s , a ) , \quad \forall ( s , a ) \neq \left( s _ { t } , a _ { t } \right) ,$$




*Pseudocode:* Policy searching by Q-learning **(on-policy version)**

---

<b>For</b> each episode, <b>do</b>
&emsp;&emsp;<b>If</b> the current st is not the target state, <b>do</b>
&emsp;&emsp;&emsp;&emsp;Collect the experience$(s_t, a_t, r_{t+1}, s_{t+1})$: In particular, take action $a_t$  following $\pi_t(s_t)$, generate $r_{t+1}$, $s_{t+1}$.
&emsp;&emsp;&emsp;&emsp;Update q-value:
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$q _ { t + 1 } \left( s _ { t } , a _ { t } \right) = q _ { t } \left( s _ { t } , a _ { t } \right) - \alpha _ { t } \left( s _ { t } , a _ { t } \right) \Big[ q _ { t } \left( s _ { t } , a _ { t } \right) - [ r _ { t + 1 } + \gamma \max _ { a \in \mathcal { A } } q _ { t } \left( s _ { t + 1 } , a \right) ]\Big] ,$
&emsp;&emsp;&emsp;&emsp;Update policy:
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\pi_{t+1}(a|s_t) = 1 - \frac{\epsilon}{|\mathcal{A}|}(|\mathcal{A}| - 1) \text{ if } a = \arg\max_a q_{t+1}(s_t, a)$
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\pi_{t+1}(a|s_t) = \frac{\epsilon}{|\mathcal{A}|} \text{ otherwise}$

---

*Pseudocode:* Policy searching by Q-learning **(off-policy version)**

---

<b>For</b> each episode $\left\{ s _ { 0 } , a _ { 0 } , r _ { 1 } , s _ { 1 } , a _ { 1 } , r _ { 2 } , \ldots \right\}$ generated by $\pi_b$, <b>do</b>

&emsp;&emsp;<b>For</b> each step $t = 0, 1, 2, \dots$ of the episode, <b>do</b>
&emsp;&emsp;&emsp;&emsp;Collect the experience$(s_t, a_t, r_{t+1}, s_{t+1})$: In particular, take action $a_t$  following $\pi_t(s_t)$, generate $r_{t+1}$, $s_{t+1}$.
&emsp;&emsp;&emsp;&emsp;Update q-value:
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$q _ { t + 1 } \left( s _ { t } , a _ { t } \right) = q _ { t } \left( s _ { t } , a _ { t } \right) - \alpha _ { t } \left( s _ { t } , a _ { t } \right) \Big[ q _ { t } \left( s _ { t } , a _ { t } \right) - [ r _ { t + 1 } + \gamma \max _ { a \in \mathcal { A } } q _ { t } \left( s _ { t + 1 } , a \right) ]\Big] ,$
&emsp;&emsp;&emsp;&emsp;Update policy:
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\pi_{T,t+1}(a|s_t) = 1 \text{ if } a = \arg\max_a q_{t+1}(s_t, a)$
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\pi_{T,t+1}(a|s_t) = 0 \text{ otherwise}$

---

where 
- $\pi_b$ is the behavior policy for generating $a_t$ in $s_t$. It can be any policy.
- $\pi_T$ is the target policy.

Q-learning aims to solve the **Bellman optimality equation**: 

$$q ( s , a ) = \mathbb { E } \left[ R _ { t + 1 } + \gamma \max _ { a } q \left( S _ { t + 1 } , a \right) \mid S _ { t } = s , A _ { t } = a \right] , \quad \forall s , a .$$ 
## <span style="color:#e5b567;">Value Function Approximation</span>

### Algorithm for state value estimation

#### Policy evaluation problem: 
Let $v_\pi(s)$ and $\hat{v}(s, w)$ be the true state value and a function for approximation. Our goal is to find an optimal parameter $w$ so that $\hat{v}(s, w)$ can best approximate $v_\pi(s)$ for every $s$. To find the optimal $w$, we need two steps:
- The first step is to define an objective function.
- The second step is to derive algorithms optimizing the objective  function.

The **objective function** is$$J ( w ) = \mathbb { E } \left[ \left( v _ { \pi } ( S ) - \hat { v } ( S , w ) \right) ^ { 2 } \right] $$
Our goal is to find the best $w$ that can minimize $J(w)$.

The expectation is with respect to the random variable $S \in \cal{S}$. What is the probability distribution of $S$?
An intuitive way is to use a **uniform distribution** by treating all the states to be equally important by setting the probability of each state as $1/|\cal{S}|$. 
- In this case, the objective function becomes
$$J ( w ) = \mathbb { E } \left[ \left( v _ { \pi } ( S ) - \hat { v } ( S , w ) \right) ^ { 2 } \right] = \frac { 1 } { | \mathcal { S } | } \sum _ { s \in \mathcal { S } } \left( v _ { \pi } ( s ) - \hat { v } ( s , w ) \right) ^ { 2 } $$
- The states may not be equally important. For example, some states may be rarely visited by a policy. Hence, this way does not consider the real dynamics of the Markov process under the given policy.

Hence introducing **stationary distribution**. In short, it describes the long-run behavior of a Markov process.

- Let $\left\{ d _ { \pi } ( s ) \right\} _ { s \in \mathcal { S } }$ denote the stationary distribution of the Markov process under policy $\pi$. By definition, $d _ { \pi } ( s ) \geq 0$ and $\sum _ { s \in \mathcal { S } } d _ { \pi } ( s ) = 1$. 
- The objective function can be rewritten as a weighted squared error.
$$J ( w ) = \mathbb { E } \left[ \left( v _ { \pi } ( S ) - \hat { v } ( S , w ) \right) ^ { 2 } \right] = \sum _ { s \in \mathcal { S } } d _ { \pi } ( s ) \left( v _ { \pi } ( s ) - \hat { v } ( s , w ) \right) ^ { 2 } $$
- Since more frequently visited states have higher values of $d_\pi(s)$, their weights in the objective function are also higher than those rarely visited states.

> Stationary distribution is also called steady-state distribution, or  limiting distribution.
> 
>  Let $n_\pi(s)$ denote the number of times that $s$ has been visited in a very long episode generated by $\pi$. Then, $d_\pi(s)$ can be approximated by
>  $$d _ { \pi } ( s ) \approx \frac { n _ { \pi } ( s ) } { \sum _ { s ^ { \prime } \in \mathcal { S } } n _ { \pi } \left( s ^ { \prime } \right) }$$

#### Optimization algorithms

While we have the objective function, the next step is to optimize it. To minimize the objective function $J(w)$, we can use the gradient-descent algorithm:$$w _ { k + 1 } = w _ { k } - \alpha _ { k } \nabla _ { w } J \left( w _ { k } \right)$$The true gradient is
$$
\begin{align}
\nabla _ { w } J ( w ) &= \nabla _ { w } \mathbb { E } \left[ \left( v _ { \pi } ( S ) - \hat { v } ( S , w ) \right) ^ { 2 } \right] \\
&= \mathbb { E } \left[ \nabla _ { w } \left( v _ { \pi } ( S ) - \hat { v } ( S , w ) \right) ^ { 2 } \right]\\
&= 2 \mathbb { E } \left[ \left( v _ { \pi } ( S ) - \hat { v } ( S , w ) \right) \left( - \nabla _ { w } \hat { v } ( S , w ) \right) \right]\\
&= - 2 \mathbb { E } \left[ \left( v _ { \pi } ( S ) - \hat { v } ( S , w ) \right) \nabla _ { w } \hat { v } ( S , w ) \right]
\end{align}
$$

 We can use the stochastic gradient to replace the true gradient:
 
$$w _ { t + 1 } = w _ { t } + \alpha _ { t } \left( v _ { \pi } \left( s _ { t } \right) - \hat { v } \left( s _ { t } , w _ { t } \right) \right) \nabla _ { w } \hat { v } \left( s _ { t } , w _ { t } \right) $$

where $s_t$ is a sample of $S$. Here, $2\alpha k$ is merged to $\alpha k$.

- This algorithm is **not implementable** because it requires the true state value $v_\pi$, which is the unknown to be estimated.
  
  Therefore, we can replace $v_\pi(s_t)$ with an approximation so that the algorithm is implementable.

In particular,

- First, **Monte Carlo learning with function approximation**: 
  
  Let $g_t$ be the discounted return starting from $s_t$ in the episode. Then, $g_t$ can be used to approximate $v_\pi(s_t)$. The algorithm becomes$$w _ { t + 1 } = w _ { t } + \alpha _ { t } \left( g _ { t } - \hat { v } \left( s _ { t } , w _ { t } \right) \right) \nabla _ { w } \hat { v } \left( s _ { t } , w _ { t } \right) $$
- Second, **TD learning with function approximation**:
  
  By the spirit of TD learning, $r _ { t + 1 } + \gamma \hat { v } \left( s _ { t + 1 } , w _ { t } \right)$ can be viewed as an approximation of $v_\pi(s_t)$. Then, the algorithm becomes$$w _ { t + 1 } = w _ { t } + \alpha _ { t } \left[ r _ { t + 1 } + \gamma \hat { v } \left( s _ { t + 1 } , w _ { t } \right) - \hat { v } \left( s _ { t } , w _ { t } \right) \right] \nabla _ { w } \hat { v } \left( s _ { t } , w _ { t } \right) $$ 

*Pseudocode:* TD learning with function approximation

<b>Initialization</b>: A function $\hat{v}(s, w)$ that is a differentiable in $w$. Initial parameter $w_0$.  

<b>Aim</b>: Approximate the true state values of a given policy $\pi$.

---
<b>For</b> each episode generated following the policy $\pi$, <b>do</b>
&emsp;&emsp;<b>For</b> each step $\left( s _ { t } , r _ { t + 1 } , s _ { t + 1 } \right)$, <b>do</b>
&emsp;&emsp;&emsp;&emsp;In the general case,
&emsp;&emsp;&emsp;&emsp;$w _ { t + 1 } = w _ { t } + \alpha _ { t } \left[ r _ { t + 1 } + \gamma \hat { v } \left( s _ { t + 1 } , w _ { t } \right) - \hat { v } \left( s _ { t } , w _ { t } \right) \right] \nabla _ { w } \hat { v } \left( s _ { t } , w _ { t } \right)$
&emsp;&emsp;&emsp;&emsp;In the linear case,
&emsp;&emsp;&emsp;&emsp;$w _ { t + 1 } = w _ { t } + \alpha _ { t } \left[ r _ { t + 1 } + \gamma \phi ^ { T } \left( s _ { t + 1 } \right) w _ { t } - \phi ^ { T } \left( s _ { t } \right) w _ { t } \right] \phi \left( s _ { t } \right)$

---

#### Selection of function approximators:

An important question that has not been answered: How to select the function $\hat{v}(s, w)$?

The first approach, which was widely used before, is to use a **linear function**.$$\hat { v } ( s , w ) = \phi ^ { T } ( s ) w$$
  - Here, $\phi(s)$ is the feature vector, which can be a polynomial basis, Fourier basis, ...
  
 -  Disadvantage: Difficult to select appropriate feature vectors.

  - Advantage: The theoretical properties of the TD algorithm in the linear case can be much better understood than in the nonlinear case.

Linear function approximation is still powerful in the sense that the **tabular representation** is merely a special case of linear function approximation:

- First, consider the special feature vector for state $s$:$$\phi ( s ) = e _ { s } \in \mathbb { R } ^ { | \mathcal { S } | } $$  where $e_s$ is a vector with the $s$th entry as $1$ and the others as $0$.
- In this case, $$\hat { v } ( s , w ) = e _ { s } ^ { T } w = w ( s ) $$ 
  where $w(s$) is the $s$th entry of $w$.
- Recall that the TD-Linear algorithm is$$w _ { t + 1 } = w _ { t } + \alpha _ { t } \left[ r _ { t + 1 } + \gamma \phi ^ { T } \left( s _ { t + 1 } \right) w _ { t } - \phi ^ { T } \left( s _ { t } \right) w _ { t } \right] \phi \left( s _ { t } \right) $$ 
  When $=\phi(s_t) = e_s$, the above algorithm becomes$$w _ { t + 1 } = w _ { t } + \alpha _ { t } \left( r _ { t + 1 } + \gamma w _ { t } \left( s _ { t + 1 } \right) - w _ { t } \left( s _ { t } \right) \right) e _ { s }$$ 
  This is a vector equation that merely updates the $s_t$th entry of $w_t$.

- Multiplying $e_{s_t}^T$ on both sides of the equation gives $$w_{t + 1}\left(s_{t}\right) = w_{t}\left(s_{t}\right) + \alpha _{t}\left(r_{t + 1} + \gamma w_{t}\left(s_{t + 1}\right)−w_{t}\left(s_{t}\right)\right)$$
  which is exactly the tabular TD algorithm.
  
The second approach, which is widely used nowadays, is to use a **neural network** as a nonlinear function approximator.
  
  - The input of the NN is the state, the output is $\hat{v}(s, w)$, and the network parameter is $w$.

### Sarsa with function approximation

So far, we merely considered the problem of state value estimation. That is we hope $\hat{v} \approx v_{\pi}$. To search for optimal policies, we need to estimate action values.

The Sarsa algorithm with value function approximation is
$$
w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t) - \hat{q}(s_t, a_t, w_t) \right] \nabla_w \hat{q}(s_t, a_t, w_t)
$$
This is the same as the algorithm introduced previously except that $\hat{v}$ is replaced by $\hat{q}$.

To search for optimal policies, we can combine policy evaluation and  policy improvement.

*Pseudocode:* Sarsa with function approximation

---
<b>Aim</b>: Search a policy that can lead the agent to the target from an initial state-action pair $(s_0, a_0)$.
<b>For</b> each episode, <b>do</b>

&emsp;&emsp;<b>If</b> the current $s_t$ is not the target state, <b>do</b>
&emsp;&emsp;&emsp;&emsp;Take action $a_t$  following $\pi_t(s_t)$, generate $r_{t+1}$, $s_{t+1}$, and then take action $a_{t+1}$ following $π_t(s_{t+1})$
&emsp;&emsp;&emsp;&emsp;**Value update (parameter update):**
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t) - \hat{q}(s_t, a_t, w_t) \right] \nabla_w \hat{q}(s_t, a_t, w_t)$
&emsp;&emsp;&emsp;&emsp;**Policy update:**
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\pi_{t+1}(a|s_t) = 1 - \frac{\varepsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)|-1) \text{ if } a = \arg\max_{a\in\mathcal{A}(s_t)} \hat{q}(s_t, a, w_{t+1})$
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$\pi_{t+1}(a|s_t) = \frac{\varepsilon}{|\mathcal{A}(s)|} \text{ otherwise}$

---

### Deep Q-learning

Deep Q-learning aims to **minimize the objective function/loss function**:
$$J ( w ) = \mathbb { E } \left[ \left( R + \gamma \max _ { a \in \mathcal { A } \left( S ^ { \prime } \right) } \hat { q } \left( S ^ { \prime } , a , w \right) - \hat { q } ( S , A , w ) \right) ^ { 2 } \right] $$ where $(S, A, R, S')$ are random variables.
- This is actually the Bellman optimality error. That is because$$q ( s , a ) = \mathbb { E } \left[ R _ { t + 1 } + \gamma \max _ { a \in \mathcal { A } \left( S _ { t + 1 } \right) } q \left( S _ { t + 1 } , a \right) \mid S _ { t } = s , A _ { t } = a \right] , \quad \forall s , a$$ The value of $R + \gamma \max _ { a \in \mathcal { A } \left( S ^ { \prime } \right) } \hat { q } \left( S ^ { \prime } , a , w \right) - \hat { q } ( S , A , w )$ should be zero in the expectation sense.

We use gradient decsent to minimize the objective function, but it is tricky because in $J(w)$, the parameter $w$ not only appears in $\hat{q}(S, A, w)$ but also in$$R + \gamma \max _ { a \in \mathcal { A } \left( S ^ { \prime } \right) } \hat { q } \left( S ^ { \prime } , a , w \right) \triangleq y$$For the sake of simplicity, we can assume that $w$ in $y$ is fixed (at least for a while) when we calculate the gradient. To do that, we can introduce two networks.

- One is a main network representing $\hat{q}(s, a, w)$
- The other is a target network $\hat{q}(s, a, w_T)$

The objective function in this case degenerates to$$J = \mathbb { E } \left[ \left( R + \gamma \max _ { a \in \mathcal { A } \left( S ^ { \prime } \right) } \color{red}{\hat { q } \left( S ^ { \prime } , a , w _ { T } \right)} - \color{#0AF}{\hat { q } ( S , A , w )} \right) ^ { 2 } \right] $$ where $w_T$ is the target network parameter.

When $w_T$ is fixed, the gradient of $J$ can be easily obtained as$$\nabla _ { w } J = \mathbb { E } \left[ \left( R + \gamma \max _ { a \in \mathcal { A } \left( S ^ { \prime } \right) } \color{red}{\hat { q } \left( S ^ { \prime } , a , w _ { T } \right)} - \color{#0AF}{\hat { q } ( S , A , w )} \right) \color{#0AF}{\nabla _ { w } \hat { q } ( S , A , w )} \right] $$The basic idea of deep Q-learning is to use the gradient-descent algorithm to minimize the objective function. **Such an optimization process evolves two important techniques that deserve special attention.**

#### Main network & Target network

Let $w$ and $w_T$ denote the parameters of the main and target networks, respectively. They are set to be the same initially. 

In every iteration, we draw a mini-batch of samples $\{(s, a, r, s')\}$ from the replay buffer (will be explained in <span style="color:#e5b567;">Experience replay</span>). The inputs of the networks include state $s$ and action $a$. The target output is$$y_T \triangleq r + \gamma \max _ { a \in \mathcal { A } \left( S ^ { \prime } \right) } \color{red}{\hat { q } \left( S ^ { \prime } , a , w_T \right)}$$Then, we directly minimize the TD error or called loss function $(y_T - \hat{q}(s,a,w))^2$ over the mini-batch $\{(s,a,y_T)\}$.
#### Experience replay

After we have collected some experience samples, we **do NOT use these samples** **in the order they were collected**. Instead, we store them in a set, called replay buffer $\mathcal{B} \triangleq \{(s,a,r,s')\}$.

Every time we train the neural network, we can draw a mini-batch of random samples from the replay buffer, so that **we can avoid the batch being composed of correlated data in the same episode and instead using i.i.d. data** for each batch during training, which is beneficial for the networks to learn properly.

The draw of samples is called **experience replay**, and it should follow a uniform distribution (in order to sample i.i.d. data mentioned above).

> To be specific, let's review the objective function:$$J = \mathbb { E } \left[ \left( R + \gamma \max _ { a \in \mathcal { A } \left( S ^ { \prime } \right) } \hat { q } \left( S ^ { \prime } , a , w _ { T } \right) - \hat { q } ( S , A , w ) \right) ^ { 2 } \right] $$
> -  $(S,A) \sim d$: $(S, A)$ is an index and treated as a single random variable
> - $R \sim p(R|S, A), \ \ \ S' \sim (S'|S,A)$: $R$ and $S$ are determined by the system model.
> - The distribution of the state-action pair $(S, A)$ is assumed **to be uniform**.
> 
> However, the samples are not uniformly collected because they are generated consequently by certain policies. **To break the correlation between consequent samples**, we can use the experience replay technique by **uniformly drawing samples from the replay buffer**.



*Pseudocode:* Deep Q-learning (off-policy version)

---

<b>Aim</b>: Learn an optimal target network to approximate the optimal action values from the experience samples generated by a behavior policy $π_b$.

Store the experience samples generated by $π_b$ in a replay buffer $\mathcal{B} = \{(s, a, r, s')\}$
&emsp;&emsp;<b>For</b> each iteration, <b>do</b>
&emsp;&emsp;&emsp;&emsp;Uniformly draw a mini-batch of samples from $\mathcal{B}$
&emsp;&emsp;&emsp;&emsp;<b>For</b> each sample $(s, a, r, s')$, calculate the target value as $y_T \triangleq r + \gamma \max _ { a \in \mathcal { A } \left( S ^ { \prime } \right) } \hat { q } \left( S ^ { \prime } , a , w_T \right)$, where $w_T$ is the parameter of the target network
&emsp;&emsp;&emsp;&emsp;Update the main network to minimize $(y_T - \hat{q}(s,a,w))^2$ using the mini-batch $\{(s,a,y_T)\}$
&emsp;&emsp;Set $w_T = w$ every $C$ iterations

---

## <span style="color:#e5b567;">Policy Gradient Methods</span>

Previously, policies have mostly been represented by tables, where each entry of the table is indexed by a state and an action, and we can directly access or change a value in the table.

Now, policies can be represented by parameterized functions (e.g. a neural network):$$\pi(a|s,\theta)$$
where $\theta \in \mathbb{R}^m$ is a parameter vector.

**Differences between tabular and function representations:**
 - **First, how to define optimal policies?**
   
   When represented as a table, a policy $\pi$ is optimal if it can maximize every state value.
   
   When represented by a function, a policy $\pi$ is optimal if it can  maximize **certain scalar metrics** (or objective functions) $J(\theta)$.
   
- **Second, how to access the probability of an action?**
  
  In the tabular case, the probability of taking $a$ at $s$ can be directly accessed by looking up the tabular policy.
  
  In the case of function representation, we need to calculate the value of $\pi(a|s, \theta)$ given the function structure and the parameter.

- **Third, how to update policies?**
  
  When represented by a table, a policy $\pi$ can be updated by directly changing the entries in the table.
  
  When represented by a parameterized function, a policy $\pi$ cannot be  updated in this way anymore. Instead, it can only be updated by  optimizing the parameter $\theta$: $$\theta _ { t + 1 } = \theta _ { t } + \alpha \nabla _ { \theta } J \left( \theta _ { t } \right)$$
### Metrics to define optimal policies

There are two metrics. The first metric is the average state value or simply called **average value**, and the second metric is average one-step reward or simply **average reward**.

#### Average value

The metric is defined as: $$\bar { v } _ { \pi } = \sum _ { s \in \mathcal { S } } d ( s ) v _ { \pi } ( s )$$where
- $\bar { v } _ { \pi }$ is a weighted average of the state values
-  $d(s) \ge 0$ is the weight for state $s$. Since $\sum _ { s \in \mathcal { S } } d ( s )=1$, we can interpret $d(s)$ as a probability distribution. Then, the metric can be written as $$\bar{v}_{\pi}=\mathbb{E}[v_{\pi}(S)]$$where $S \sim d$.

**How to select the distribution $d$? There are two cases.**

The first case is that $d$ is independent of the policy $\pi$.

- In this case, we specifically denote $d$ as $d_0$ and $\bar{v}_{\pi}$ as $\bar{v}_{\pi}^0$.
- How to select $d_0$? One trivial way is to treat all the states equally important and hence select $d_0(s) = 1/|\mathcal{S}|$.
  
  Another important case is that we are only interested in a specific  state $s_0$. For example, the episodes in some tasks always start from  the same state $s_0$. Then, we only care about the long-term return  starting from $s_0$. In this case,$$d_0(s_0)=1, \ d_0(s \neq s_0)=0$$
The first case is that $d$ depends on the policy $\pi$.

- A common way to select $d$ as $d_\pi(s)$, which is the **stationary distribution** under $\pi$. 
- The interpretation of selecting $d_\pi$ is as follows:
  
  If one state is frequently visited in the long run, it is more important and deserves more weight.
  
  If a state is hardly visited, then we give it less weight.

#### Average reward

In particular, the metric is$$\bar { r } _ { \pi } \triangleq \sum _ { s \in \mathcal { S } } d _ { \pi } ( s ) r _ { \pi } ( s ) = \mathbb { E } \left[ r _ { \pi } ( S ) \right]$$ where $S \sim d_\pi$. Here,$$r _ { \pi } ( s ) \triangleq \sum _ { a \in \mathcal { A } } \pi ( a | s ) r ( s , a )$$is the average of the one-step immediate reward that can be obtained starting from state $s$, and$$r ( s , a ) = \mathbb { E } [ R | s , a ] = \sum _ { r } r p ( r | s , a )$$
- The weight $d_\pi$ is the stationary distribution.
- As its name suggests, $\bar { r } _ { \pi }$ is simply a weighted average of the one-step immediate rewards.

An important property is that$$\begin{align}
\lim _ { n \rightarrow \infty } \frac { 1 } { n } \mathbb { E } \left[ \sum _ { k = 1 } ^ { n } R _ { t + k } \mid S _ { t } = s _ { 0 } \right] &= \lim _ { n \rightarrow \infty } \frac { 1 } { n } \mathbb { E } \left[ \sum _ { k = 1 } ^ { n } R _ { t + k } \right]\\
&= \sum _ { s } d _ { \pi } ( s ) r _ { \pi } ( s )\\
&= \bar { r } _ { \pi }
\end{align}
$$Note that the LHS is exactly the average single-step reward $(R_{t+1},R_{t+2},\dots)$ along a trajectory generated by the agent following a given policy.
 
>**Remarks about the metrics:**
>- All these metrics are functions of $\pi$. Since $π$ is parameterized by $\theta$, these metrics are functions of $\theta$. Therefore, we can search for the optimal values of $\theta$ to maximize these metrics. This is the basic idea of policy gradient methods.
>- Intuitively, $\bar { r } _ { \pi }$ is more short-sighted because **it merely considers the  immediate rewards, whereas $\bar { v } _ { \pi }$ considers the total reward overall steps**.
>- However, the two metrics are equivalent to each other. In the discounted case where $\gamma < 1$, it holds that$$\bar { r } _ { \pi }=(1-\gamma)\bar { v } _ { \pi }$$
### Gradients of the metrics

Summary of the results about the gradients:$$\nabla _ { \theta } J ( \theta ) = \sum _ { s \in \mathcal { S } } \eta ( s ) \sum _ { a \in \mathcal { A } } \nabla _ { \theta } \pi ( a \mid s , \theta ) q _ { \pi } ( s , a )$$where
- $J(\theta)$ can be $\bar { v } _ { \pi }$, $\bar { r } _ { \pi }$, or $\bar { v } _ { \pi }^0$
- "$=$" may denote "strict equality", "approximation", or "proportional to"
- $\eta$ is a distribution or weight of the states

Some specific results:$$\begin{align}\nabla _ { \theta } \bar { r } _ { \pi } &\simeq \sum _ { s } d _ { \pi } ( s ) \sum _ { a } \nabla _ { \theta } \pi ( a \mid s , \theta ) q _ { \pi } ( s , a ) \\
\nabla _ { \theta } \bar { v } _ { \pi } &= \frac { 1 } { 1 - \gamma } \nabla _ { \theta } \bar { r } _ { \pi }\\
\nabla _ { \theta } \bar { v } _ { \pi } ^ { 0 } &= \sum _ { s \in \mathcal { S } } \rho _ { \pi } ( s ) \sum _ { a \in \mathcal { A } } \nabla _ { \theta } \pi ( a \mid s , \theta ) q _ { \pi } ( s , a )
\end{align}$$**A compact and useful form of the gradient:**$$\begin{align}
\nabla _ { \theta } J ( \theta ) &= \sum _ { s \in \mathcal { S } } \eta ( s ) \sum _ { a \in \mathcal { A } } \nabla _ { \theta } \pi ( a \mid s , \theta ) q _ { \pi } ( s , a )\\
&= \mathbb { E } \left[ \nabla _ { \theta } \ln \pi ( A \mid S , \theta ) q _ { \pi } ( S , A ) \right] 
\end{align}
$$where $S \sim \eta$ and $A \sim \pi(A | S, \theta)$
Advantage of this form: we can use samples to **approximate the gradient**.

>**Remarks:** 
>- Because we need to calculate $\ln \pi(a|s, \theta)$, we must ensure that for all $s, a, \theta$$$\pi(a|s)>0$$This can be archived by applying softmax to normalize the entries in a vector.
>  
>- Such a form based on the softmax function can be realized by a neural network whose input is $s$ and parameter is $\theta$. The network has $|\mathcal{A}|$  outputs, each of which corresponds to $\pi(a|s, \theta)$ for an action $a$. The  activation function of the output layer should be softmax.
>  
>- Since $\pi(a|s, \theta) > 0$ for all $a$, the parameterized policy is stochastic and  hence exploratory. But there also exist **deterministic policy gradient (DPG)** methods.


### Gradient-ascent algorithm (REINFORCE)

