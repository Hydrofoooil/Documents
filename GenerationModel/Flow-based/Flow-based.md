## Normalizing Flow

**Generator:**

A generator is a network which defines a probability distribution $P_G$.

<img src="截图 2025-12-17 23-15-01.png" style="zoom:70%;" />

$$
G^* = \arg \max \limits_{G} \sum_{i=1}^{m} logP_G(x^i) \approx \arg \min \limits_{G} KL(P_{data}||P_G)
$$
Flow-based model directly optimizes the objective function.

**Change of Variable formulation:**

Given $x = f(z)$：
$$
p \left( x ^ { \prime } \right) \left| \operatorname { d e t } \left( J _ { f } \right) \right| = \pi \left( z ^ { \prime } \right) \\
p \left( x ^ { \prime } \right) = \pi \left( z ^ { \prime } \right) \left| \operatorname { d e t } \left( J _ { f ^ { - 1 } } \right) \right|
$$
where 

- $p$ is the distribution of $x$ and $\pi$ is the distribution of $z$

- $x'=f(z')$  and $f$ has to be irreversible.
- $\operatorname { d e t } \left( J _ { f ^ { - 1 } } \right) = \frac{1}{\operatorname { d e t } \left( J _ { f } \right) }$

**In our case:**
$$
z^i = G^{-1}(x^i) \\
p_G \left( x ^ { i } \right) = \pi \left( z ^ { i } \right) \left| \operatorname { d e t } \left( J _ { G ^ { - 1 } } \right) \right|
$$
**Actually, we train $G^{-1}$, but use $G$ for generation.**

If we train $G^{-1}$ to map $P_{data}$ into normal distribution, then $G$ is able to inversely map the normal distribution into $P_{data}$.

<img src="截图 2025-12-17 23-57-06.png" style="zoom: 70%;" />

$$
\begin{align*}
p_1(x^i) &= \pi(z^i) \left( \left| \det(J_{G_1^{-1}}) \right| \right) \\
p_2(x^i) &= \pi(z^i) \left( \left| \det(J_{G_1^{-1}}) \right| \right) \left( \left| \det(J_{G_2^{-1}}) \right| \right) \\
&\vdots \\
p_K(x^i) &= \pi(z^i) \left( \left| \det(J_{G_1^{-1}}) \right| \right) \cdots \left( \left| \det(J_{G_K^{-1}}) \right| \right) \\
\log p_K(x^i) &= \log \pi(z^i) + \sum_{h=1}^{K} \log \left| \det(J_{G_h^{-1}}) \right|
\end{align*}
$$
<img src="截图 2025-12-18 00-10-57.png" style="zoom:70%;" />

Intuitively, to train $G^{-1}$ to map $P_{data}$ into normal distribution, our goal seems to be making $z^i$ be as close to zero vector (i.e. the mean of normal distribution) as possible. However, if $z^i$ is always zero, $J_{G_h^{-1}}$ would be zero matrix and thus $\det{J_{G_h^{-1}}}$ would be `-inf`.

Therefore we need to make some kind of trade-off in between.

**What do we actually do?**

### Coupling Layer

  $\color{lime}{\text{ (NICE, Real NVP)}}$

<img src="截图 2025-12-18 00-30-31.png" alt="截图 2025-12-18 00-30-31" style="zoom:70%;" />

where F and H is learned during training and can be any function that don't necessarily have to be irreversible.

As shown in the picture, we can compute $z_i$ given $x_i$, so we can compute $G^{-1}$.

Then how to compute $\operatorname { d e t } \left( J _ {G } \right)$ ?

<img src="截图 2025-12-18 00-44-05.png" alt="截图 2025-12-18 00-44-05" style="zoom:70%;" />

In fact, $J _ {G}$ is an *lower triangular matrix*! 

Therefore, 
$$
\begin{align}
\operatorname { d e t } \left( J _ {G } \right)&=  \frac { \partial x _ { d + 1 } } { \partial z _ { d + 1 } } \frac { \partial x _ { d + 2 } } { \partial z _ { d + 2 } } \cdots \frac { \partial x _ { \mathrm { D } } } { \partial z _ { \mathrm { D } } } \\
&= \beta_{d+1}\beta_{d+2}\cdots\beta_{D}
\end{align}
$$

> **Why Coupling Layers require a "copy" part?**
>
> The "copy" part serves as a mathematical "cheat code" to satisfy two critical constraints simultaneously:
>
> - Guarantees Invertibility:
>
>   By keeping one half of the data unchanged ($z_1, \cdots,z_d = x_1, \cdots,x_d$), we break the dependency cycle during the reverse pass. Since we know $z_1, \cdots,z_d$, we immediately know $x_1, \cdots,x_d$, which allows us to recompute F and H needed to recover the other half ($x_{d+1}, \cdots,x_D$). Without this, inversion would be mathematically intractable.
>
> - Efficient Computation:
>
>   It forces the Jacobian matrix to be *lower triangular matrix*. This allows $\operatorname { d e t } \left( J _ {G } \right)$ to be calculated with $O(N)$ complexity rather than a computationally expensive matrix operation $O(N^3)$ .

#### Coupling Layer - Stacking

<img src="截图 2025-12-18 00-50-42.png" alt="截图 2025-12-18 00-50-42" style="zoom: 70%;" />

**Image generation case:**

<img src="截图 2025-12-18 00-55-57.png" alt="截图 2025-12-18 00-55-57" style="zoom:70%;" />

- <b>Spatial checkerboard pattern masking:</b> Selecting different groups of pixels to *copy*(shallow green) or *transform*(deep green) according to `x_index` and `y_index` in each layer.

- <b>Channel-wise masking:</b> Selecting channels to *copy*(shallow green) or *transform*(deep green) in each layer.

- $1 \times 1$ <b>Convolution:</b>

  $\color{lime}{\text{(Glow, 2018)}}$

  <img src="截图 2025-12-18 01-27-42.png" alt="截图 2025-12-18 01-27-42" style="zoom: 40%;" />

Note that $W$ is a learned matrix so the possibility of it being invertible (i.e. $\det{W}$ is exactly $0$) is near zero.
$$
x = f(z) = Wz
$$
In fact, $J_f = W$! Therefore, 

<img src="截图 2025-12-18 01-28-51.png" alt="截图 2025-12-18 01-28-51" style="zoom:70%;" />

where the image shape is $d \times d$.

### Autoregressive Flow

- $\color{lime}{\text{(Inverse AF)}}$
- $\color{lime}{\text{(Neural AF)}}$
- $\color{lime}{\text{(Masked AF)}}$

### Applications

<img src="截图 2025-12-18 01-42-12.png" alt="截图 2025-12-18 01-42-12" style="zoom: 70%;" />

<img src="截图 2025-12-18 01-41-58.png" alt="截图 2025-12-18 01-41-58" style="zoom:70%;" />

## Continuous Normalizing Flow

### Flow-matching

Trajectory $X=\{x_0,\cdots,x_1\}$：$x_t$ is the position at timestep $t$ where $t\in[0,1]$.

Vector Field $u_t$: Defines the movement at timestep $t$ by giving the current speed.
$$
\displaylines{
u_t(x_t)=v=\frac{dx_t}{dt} \\
\left.
\begin{array}{c}
    x_0 \\
    u \\
    t: [0,1]
\end{array}
\right\}
\quad
X: \{x_0, \dots, x_1\}
}
$$
Flow $\psi$: A set of trajectories.
$$
\displaylines{
x_t = \psi_t(x_0) \\
\frac{dx_t}{dt} = u_t(x_t) \\
\Rightarrow \frac{d\psi_t(x_0)}{dt} = u_t(\psi_t(x_0))
}
$$

#### Inference

*Pseudocode：*

<b>Initialization:</b> Learn $u_t^\theta$ with a neural network, and set total timestep $n$.

---

Set t = 0 and $h=\frac{1}{n}$
Sample $x_0$ from $p_{init}$
<b>for</b> i = 1, ... , n-1 <b>do</b>

​	$x_{t+h}=x_t+hu_t^\theta(x_t)$

​	$t=t+h$

<b>return</b> $x_1$

---

#### Training

Define *Mean Square Error* loss as: $L ( \theta ) = \left\| u _ { t } ^ { \theta } \left( x _ { t } \right) - u _ { t } ^ { \text{target} } \left( x _ { t } \right) \right\| ^ { 2 }$ where the target flow $u _ { t } ^ { \text{target} }$ is defined by ourselves.

First define a <b>conditional</b> probability trajectory $p_t(\cdot | z )$ that satisfies follow conditions:
$$
\displaylines{
p_0(\cdot | z ) = N(0,1) \\
p_1(\cdot | z ) = N(z,0)
}
$$
where $z$ is sampled from training data, and for each $z$, $x_0$ is randomly sampled from normal distribution. 

<img src="截图 2025-12-20 02-12-42.png" alt="截图 2025-12-20 02-12-42" style="zoom:70%;" />

In most cases the trajectory is defined as $x_t = (1-t)x_0 + t z$ because the "straight path" is considered as the optimal transport.

Then we can build a target flow (conditional, given $z$) $\psi _ { t } ^ { \text{target} } \left( x _ { 0 } | z \right)$ that analogously satisfies follow conditions: 
$$
\displaylines{
\psi _ { 0 } ^ { \text{target} } \left( x _ { 0 } | z \right) = x_0 \\
\psi _ { 1 } ^ { \text{target} } \left( x _ { 0 } | z \right) = z
}
$$
Next, compute the vector field $u_t(x_t|z)$ which is covered by $\psi _ { t } ^ { \text{target} }$: 
$$
\frac{d\psi_t(x_0|z)}{dt} = u_t(\psi_t(x_0|z)|z)
$$
Finally, we can use MSE for *Conditional Flow Matching*: $L _{CFM}( \theta ) = \left\| u _ { t } ^ { \theta } \left( x _ { t } \right) - u _ { t } ^ { \text{target} } \left( x _ { t } |z\right) \right\| ^ { 2 }$ as loss function to train our flow-matching model.

The intuition is that at each iteration, the neural network learns a part of vector field which is corresponding to the data from current batch. As the batches cover the entire training set, the vector field learned by the network will ultimately covers the entire training data. 

<img src="截图 2025-12-20 02-49-36.png" alt="截图 2025-12-20 02-49-36" style="zoom:70%;" />

If we run for enough epoch, the vector field will <b>generalize</b> from $u_t(x_t|\text{training data})$ to $u_t(x_t)$.

*Pseudocode：*

<b>for</b> each iteration <b>do</b>

​	Randomly sample a picture $z$ from training set

​	Randomly sample a timestep $t\in[0,1]$

​	Randomly sample a noise $\varepsilon$ as $x_0$, $\varepsilon\sim N(0,1)$

​	Compute $x_t = \psi_t^{target}(x_0)$

​	Compute $u _ { t } ^ { \theta } \left( x _ { t } \right)$ as prediction for the speed at $t$

​	Compute $u_t(x_t|z)$  as the labeled speed at $t$

​	Compute MSE loss between the labeled speed and the prediction, then update $\theta$

> Note that the advantage of Flow-matching over DDPM is that Flow-matching requires significantly fewer sampling steps than DDPM. This is because it abandons the tortuous and highly stochastic diffusion-denoising paths characteristic of DDPM, and instead learns a smooth, deterministic Optimal Transport path from noise to data. This straight 'highway' allows the sampling process to advance with much larger step sizes, enabling it to reach the destination in fewer steps and greatly improving generation efficiency.

code：

```python
def flow_matching_inference(action_expert, observation_embedding, horizon=50, steps=10):
    """
    Args:
        action_expert: 训练好的动作专家模型 (PyTorch Module)
        observation_embedding: VLM Backbone 提取的上下文特征 (Batch, Seq_Len, Dim)
        horizon: 动作预测长度 (H=50)
        steps: 积分步数 (N=10)
    """
    batch_size = observation_embedding.shape[0]
    action_dim = 14 # 假设动作维度

    # 1. 初始化：从标准正态分布采样纯噪声 x_0
    # 注意：这里的 Shape 包含了 Horizon 维度，这就是"并行"的关键
    # 我们一次性初始化了未来 50 步的所有动作
    x_t = torch.randn(batch_size, horizon, action_dim, device=device) 
    
    # 2. 定义时间步长 dt
    dt = 1.0 / steps
    
    # 3. ODE 积分循环 (从 t=0 到 t=1)
    # 这个循环是串行的，但循环内部对 Horizon 的处理是并行的
    for i in range(steps):
        # 当前时间 t (归一化到 [0, 1])
        t_value = i / steps
        
        # 构造时间张量 (Batch,) 或 (Batch, 1)
        t_tensor = torch.full((batch_size,), t_value, device=device)
        
        # 4. 模型前向传播：预测向量场 (Velocity)
        # 输入：当前带噪轨迹 x_t, 观测条件 cond, 当前时间 t
        # 输出：向量场 v (Batch, Horizon, Action_Dim)
        # 这里模型同时“看到”了这 50 步的所有状态，并计算出它们该往哪里“流”
        velocity = action_expert(x_t, observation_embedding, t_tensor)
        
        # 5. 更新状态 (Euler Step)
        # x_{t+1} = x_t + v(x_t, t) * dt
        x_t = x_t + velocity * dt
        
    # 积分结束，x_t 即为生成的去噪后的动作序列 a_{t:t+H}
    return x_t
```

The `action_expert` in the code is the training object.