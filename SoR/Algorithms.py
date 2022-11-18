import numpy as np
import matplotlib.pyplot as plt

# Define the class for Bregman SoR method


class Bregman_SoR:
    """ parrent class for SOR algorithms, based on the risk-averse of quadratic function, provide function and gradient estimators for this specific problem and Bregman method

    Attributes:
        A: samples of A_xi
        batch_size: batch size for inner and outer function value/gradient samples
        max_iter: maximum iteration number of the algorithm
        x_init: initial x
        k1, k2: coefs of h_g and h_f
        tau: step size
        beta: moving average coef
        R: radius of feasible set
        d: length of variable x
        n: number of samples of A_xi
        A_avg: E[A_xi]

    Result Attributes:
        x_traj: trajectory of x
        x_hat_traj: trajectory of x_hat
        val_F_traj: trajectory of deterministic function value
        Dh1, Dh2: D_h(x_hat^{k+1}, x_hat^k)
        Dh1_avg, Dh1_x_avg: 1/k sum_0^k D_h(x_hat^{k+1}, x_hat^k)

    """

    def __init__(self,  A, batch_size, x_init, k1, k2, tau, beta, max_iter, R, lmbda=1):
        self.A = A
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.x_init = x_init  # initial x
        self.k1 = k1
        self.k2 = k2
        self.tau = tau
        self.beta = beta
        self.lmbda = lmbda
        self.R = R

        self.d = A.shape[0]
        self.n = A.shape[2]  #
        self.A_avg = self.A.mean(axis=2)

        self.x_traj = np.zeros([self.d, self.max_iter])  # traj of x
        # traj of x_hat(calculated based on determinastic function)
        self.x_hat_traj = np.zeros([self.d, self.max_iter])
        # traj of determinastic function value
        self.val_F_traj = np.zeros(self.max_iter)
        # determinastic gradient and sample gradient
        self.grad_Fdet_traj = np.zeros([self.d, self.max_iter])
        self.grad_F_traj = np.zeros([self.d, self.max_iter])
        # D_h1,2(x_hat^{k+1}- x_hat^k)
        self.Dh1_traj, self.Dh2_traj = [], []
        # average of all previous iterations
        self.Dh1_avg_traj, self.Dh2_avg_traj = [], []

    def _sample_A(self):
        # sample the A mattrix
        idx_sample = np.random.randint(0, self.n, self.batch_size)
        return self.A[:, :, idx_sample]

    def _get_val_g(self, A_sample, x):
        # get sample value of inner function g
        g1 = 1/2 * x.T @ np.mean(A_sample, 2) @ x
        g2 = 1/2 * np.einsum('j, ijk, i', x, A_sample, x)
        g2 = np.mean(g2**2)

        return np.array([g1, g2])  # g value

    def _get_val_F(self, x):
        # get deterministic function value at x
        temp = np.einsum('j,ijk,i', x, self.A, x)
        temp = np.mean(temp**2)

        return -1/2 * x.T @ self.A_avg @ x + self.lmbda*(temp - 1/4*(x.T @ self.A_avg @ x)**2)

    def _get_grad_h(self, x):
        # get gradient of geneartining function h(x)
        return self.k1 * x + self.k2 * np.linalg.norm(x)**2 * x

    def _get_grad_g(self, A_sample, x):
        # get sample gradient of inner function g
        grad_g1 = np.einsum('ijk,j->ik', A_sample, x)
        grad_g21 = np.einsum('j,ijk,i', x, A_sample, x)
        grad_g2 = grad_g21 * grad_g1

        return np.row_stack([grad_g1.mean(axis=1), grad_g2.mean(axis=1)])

    def _get_grad_f(self, u):
        # get gradient of outter function f(which is deterministic )
        return np.array([-1 - 2 * self.lmbda * u[0], self.lmbda])

    def _solve_Breg_sub(self, x, w):
        # solve the Bregman subproblem
        grad_h = self._get_grad_h(x)
        t = self.tau * w - grad_h
        if np.all(t == 0):
            y = t
        else:
            theta = np.roots(
                [self.k2 * np.linalg.norm(t)**2, 0, self.k1, -1])
            theta = theta[np.isreal(theta)]
            theta = np.real(theta.max())
            y = -theta * t
            y = y * min(1, self.R/np.linalg.norm(y))

        return y

    def _update_u(self, A_sample, u, x_pre, x):
        g_pre = self._get_val_g(A_sample, x_pre)
        g = self._get_val_g(A_sample, x)

        u = (1 - self.beta) * (u + g - g_pre) + self.beta * g

        return u

    def _get_grad_Fdet(self, x):
        val_g = self._get_val_g(self.A, x)
        v_det = self._get_grad_g(self.A, x)
        s_det = self._get_grad_f(val_g)
        grad_Fdet = v_det.T @ s_det
        return grad_Fdet

    def train(self):
        x = self.x_init

        # initial sample
        A_sample = self._sample_A()
        u = self._get_val_g(A_sample, x)

        A_sample = self._sample_A()
        v = self._get_grad_g(A_sample, x)
        s = self._get_grad_f(u)
        w = v.T @ s

        grad_Fdet = self._get_grad_Fdet(x)
        x_hat = self._solve_Breg_sub(x, grad_Fdet)

        # save initial information
        self.x_traj[:, 0] = x
        self.x_hat_traj[:, 0] = x_hat
        self.val_F_traj[0] = self._get_val_F(x)
        self.grad_Fdet_traj[:, 0] = grad_Fdet
        self.grad_F_traj[:, 0] = w

        for iter in range(1, self.max_iter):
            x_pre = x
            # update
            x = self._solve_Breg_sub(x, w)

            A_sample = self._sample_A()
            u = self._update_u(A_sample, u, x_pre, x)

            A_sample = self._sample_A()
            v = self._get_grad_g(A_sample, x)
            s = self._get_grad_f(u)
            w = v.T @ s
            grad_Fdet = self._get_grad_Fdet(x)
            x_hat = self._solve_Breg_sub(x, grad_Fdet)

            # save information
            self.x_traj[:, iter] = x
            self.x_hat_traj[:, iter] = x_hat
            self.val_F_traj[iter] = self._get_val_F(x)
            self.grad_Fdet_traj[:, iter] = grad_Fdet
            self.grad_F_traj[:, iter] = w

    def calculate_Dh(self):
        # calculate Dh_1 and Dh_2 for x_hat
        self.Dh1_traj = self.x_hat_traj - self.x_traj
        self.Dh1_traj = np.sum(self.Dh1_traj * self.Dh1_traj, axis=0)

        self.Dh2_traj = np.zeros(self.max_iter)

        for iter in range(0, self.max_iter):
            y = self.x_hat_traj[:, iter]
            x = self.x_traj[:, iter]

            self.Dh2_traj[iter] = (np.linalg.norm(
                y)**4 / 4 - np.linalg.norm(x)**4 / 4 - np.linalg.norm(x)**2 * x @ (y-x))

    def calculate_Dh_avg(self):
        self.calculate_Dh()

        self.Dh1_avg_traj = np.zeros_like(self.Dh1_traj)
        self.Dh2_avg_traj = np.zeros_like(self.Dh2_traj)

        for i in range(self.max_iter):
            self.Dh1_avg_traj[i] = np.mean(self.Dh1_traj[:i + 1])
            self.Dh2_avg_traj[i] = np.mean(self.Dh2_traj[:i + 1])

    def plot(self, k1, k2, tau, avg=True):
        # if avg = True, plot the averaged D_h, if = False, plot each iteration

        fig, axs = plt.subplots(2, 3, figsize=(9, 8))
        fig.suptitle(f"k1={k1: .2e}, k2 = {k2: .2e}")

        if avg == True:
            if len(self.Dh1_avg_traj) == 0:
                self.calculate_Dh_avg()
            axs[0, 0].plot(k1 * self.Dh1_avg_traj / tau**2)
            axs[0, 0].set_ylabel(r"$E[D_{h_1}(\hat{x}^{k+1}, x^k)/\tau^2]$")

            axs[0, 1].plot(k2 * self.Dh2_avg_traj / tau**2)
            axs[0, 1].set_ylabel(r"$E[D_{h_2}(\hat{x}^{k+1}, x^k)/\tau^2]$")

            axs[0, 2].plot((k1 * self.Dh1_avg_traj / tau**2 +
                           k2 * self.Dh2_avg_traj / tau**2))
            axs[0, 2].set_ylabel(r"$E[D_{h}(\hat{x}^{k+1}, x^k)/\tau^2]$")

        else:
            if len(self.Dh1_traj) == 0:
                self.calculate_Dh()
            axs[0, 0].plot(k1 * self.Dh1_traj / tau**2)
            axs[0, 0].set_ylabel(r"$D_{h_1}(\hat{x}^{k+1}, x^k)/\tau^2$")

            axs[0, 1].plot(k2 * self.Dh2_traj / tau**2)
            axs[0, 1].set_ylabel(r"$D_{h_2}(\hat{x}^{k+1}, x^k)/\tau^2$")

            axs[0, 2].plot((k1 * self.Dh1_traj / tau**2 +
                           k2 * self.Dh2_traj / tau**2))
            axs[0, 2].set_ylabel(r"$D_{h}(\hat{x}^{k+1}, x^k)/\tau^2$")
        axs[1, 0].plot(self.val_F_traj)
        axs[1, 0].set_ylabel(r"$F(x^k)$")
        axs[1, 0].set_ylim(1e0, 1e6)
        axs[1, 1].plot(np.linalg.norm(self.grad_F_traj, ord=2, axis=0)**2)
        axs[1, 1].set_ylabel(r"$\|w^k\|^2$")
        axs[1, 2].plot(np.linalg.norm(self.grad_Fdet_traj, ord=2, axis=0)**2)
        axs[1, 2].set_ylabel(r"$\|\nabla F(x^k)\|^2$")
        for ax in axs.flat:
            ax.set_xlabel("iteration")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.5)
        fig.tight_layout()

        # fig.show()


class SCSC_SoR(Bregman_SoR):
    def __init__(self,  A, batch_size, x_init, alpha, beta, max_iter, R, lmbda):
        # default setting for x_hat calculation
        k1, k2, tau = 63.1, 1.58, 0.025
        self.alpha = alpha
        super().__init__(A, batch_size, x_init, k1, k2, tau, beta, max_iter, R, lmbda)

    def _projectd_gradient_step(self, x, w):
        # gradient step
        y = x - self.alpha * w
        # projection
        y = y * min(1, self.R/np.linalg.norm(y))
        return y

    def train(self):
        x = self.x_init

        # initial sample
        A_sample = self._sample_A()
        u = self._get_val_g(A_sample, x)

        A_sample = self._sample_A()
        v = self._get_grad_g(A_sample, x)
        s = self._get_grad_f(u)
        w = v.T @ s

        grad_Fdet = self._get_grad_Fdet(x)
        x_hat = self._solve_Breg_sub(x, grad_Fdet)

        # save initial information
        self.x_traj[:, 0] = x
        self.x_hat_traj[:, 0] = x_hat
        self.val_F_traj[0] = self._get_val_F(x)
        self.grad_Fdet_traj[:, 0] = grad_Fdet
        self.grad_F_traj[:, 0] = w

        for iter in range(1, self.max_iter):
            x_pre = x
            # update
            x = self._projectd_gradient_step(x, w)

            A_sample = self._sample_A()
            u = self._update_u(A_sample, u, x_pre, x)

            A_sample = self._sample_A()
            v = self._get_grad_g(A_sample, x)
            s = self._get_grad_f(u)
            w = v.T @ s

            grad_Fdet = self._get_grad_Fdet(x)
            x_hat = self._solve_Breg_sub(x, grad_Fdet)

            # save information
            self.x_traj[:, iter] = x
            self.x_hat_traj[:, iter] = x_hat
            self.val_F_traj[iter] = self._get_val_F(x)
            self.grad_Fdet_traj[:, iter] = grad_Fdet
            self.grad_F_traj[:, iter] = w


class NASA_SoR(Bregman_SoR):
    def __init__(self,  A, batch_size, x_init, tau, beta, a, b, max_iter, R, lmbda):
        # default setting for x_hat calculation

        super().__init__(A, batch_size, x_init, 63.1, 1.58, 0.025, 0.5, max_iter, R, lmbda)
        self.tau_NASA = tau
        self.beta_NASA = beta
        self.a = a
        self.b = b

    def _projectd_gradient_step(self, x, w):
        # gradient step
        y = x - 1 / self.beta_NASA * w
        # projection
        y = y * min(1, self.R/np.linalg.norm(y))
        return y

    def _update_u(self, A_sample, u, x):
        # update inner function value estimator
        g = self._get_val_g(A_sample, x)
        u = (1 - self.b * self.tau_NASA) * u + self.b * self.tau_NASA * g
        return u

    def train(self):
        x = self.x_init

        # initial sample
        A_sample = self._sample_A()
        u = self._get_val_g(A_sample, x)

        A_sample = self._sample_A()
        v = self._get_grad_g(A_sample, x)
        s = self._get_grad_f(u)
        w = v.T @ s

        grad_Fdet = self._get_grad_Fdet(x)
        x_hat = self._solve_Breg_sub(x, grad_Fdet)

        # save initial information
        self.x_traj[:, 0] = x
        self.x_hat_traj[:, 0] = x_hat
        self.val_F_traj[0] = self._get_val_F(x)
        self.grad_Fdet_traj[:, 0] = grad_Fdet
        self.grad_F_traj[:, 0] = w

        for iter in range(1, self.max_iter):
            x_pre = x
            # update

            y = self._projectd_gradient_step(x, w)
            if iter == 1:
                x = y
            else:
                x = x_pre + self.tau_NASA * (y - x_pre)

            s = self._get_grad_f(u)
            A_sample = self._sample_A()
            J = self._get_grad_g(A_sample, x)
            w = (1 - self.a * self.tau_NASA) * w + \
                self.a * self.tau_NASA * J.T @ s

            u = self._update_u(A_sample, u, x)

            grad_Fdet = self._get_grad_Fdet(x)
            x_hat = self._solve_Breg_sub(x, grad_Fdet)

            # save information
            self.x_traj[:, iter] = x
            self.x_hat_traj[:, iter] = x_hat
            self.val_F_traj[iter] = self._get_val_F(x)
            self.grad_Fdet_traj[:, iter] = grad_Fdet
            self.grad_F_traj[:, iter] = w
