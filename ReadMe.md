# Explain the algorithm detail

## SOR(Risk-Averse Problem)

Objective function is 
$$
\min_{x \in \cal{X}} - \mathbb{E}[r_\xi(x)] + \lambda \mathbb{E}[r_\xi(x)^2] - \mathbb{E}^2[r_\xi(x)], \quad \text{with} \quad r_\xi = \frac{1}{2}x^\intercal A_\xi x, \quad \cal{X} = \{\|x\| \leq R \}
$$
The inner function $g(x)$ and outer function $f(u)$ for this problem is defined as 
$$
g(x):\R^n \rightarrow \R^2 = \mathbb{E}[r_\xi(x) ; \quad r_\xi(x)^2] \\
f(u_1,u_2):\R^2 \rightarrow \R = -u_1 + \lambda u_2 - \lambda u_1^2
$$


The generating function is 
$$
\begin{align*}
h(x) &= \frac{c_1 + c_2 \|A\|}{2}\|x\|^2 + \frac{3c_2\mathbb{E}[\|A_\xi\|^2]}{4}\|x\|^4 \\
	& \triangleq \frac{k_1}2\|x\|^2 + \frac{k_2}{4}\|x\|^4
\end{align*}
$$
the gradient is 
$$
\grad h(x) = k_1 x + k_2\|x\|^2x
$$
At each iteration, the Bregman subproblem is
$$
&\arg\min _{y \in \cal{X}} \langle w^k, y - x^k \rangle + \frac{1}{\tau_k}D_h(y,x^k) \\
=&\arg\min_{y \in \cal{X}} \langle  \tau_k w^k- \grad h(x^k), y  \rangle + h(y) \\
=&\arg\min_{y \in \cal{X}} \langle t,y \rangle + \frac{k_1}{2}\|y\|^2 + \frac{k_2}{4}\|y\|^4
$$
where we define $\tau_k w^k - \grad h(x^k) \triangleq t$. (Following Lu'18) By the first-order optimality condition, 
$$
t + (k_1 + k_2\|y\|^2)y =0
$$
Hence $t $ and $y$ should have opposite directions, which is $y = -\theta t$ for some $\theta\geq 0$. If $t=0$, then $y =0$ will be the solution. When $t \neq 0$, replace $y$ with $-\theta t$ in the equation above
$$
t(1-k_1\theta - k_2 \|t\|^2 \theta^3) =0
$$
So we only need to find $\theta$ that is the root of polynomial $1-k_1\theta - k_2\|t\|^2 \theta^3=0$
