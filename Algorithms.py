import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Define the class for Bregman SoR method


class Bregman_SoR:
    """ parrent class for SOR algorithms, based on the risk-averse of quadratic function

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
        Dh1, Dh2: D_h(x_hat^{k+1}, x^k)
        Dh1_avg, Dh1_x_avg: 1/k sum_0^k D_h(x_hat^{k+1}, x^k)
        val_F_traj, val_F_avg_traj: trajectory of deterministic function value
        grad_Fdet_traj, grad_F_traj: trajectory of deterministic gradient and estimated gradient
        norm_gradFdet_traj, norm_gradFdet_avg_traj: (averaged) norm of deterministic gradient
        self.norm_gradF_traj, norm_gradF_avg_traj: (averaged) norm of estimated gradient

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
        self.val_F_avg_traj = []
        # determinastic gradient and sample gradient
        self.grad_Fdet_traj = np.zeros([self.d, self.max_iter])
        self.grad_F_traj = np.zeros([self.d, self.max_iter])
        self.norm_gradFdet_traj = []
        self.norm_gradF_traj = []
        self.norm_gradFdet_avg_traj = []
        self.norm_gradF_avg_traj = []

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
        # temp = 1/2*np.einsum('j,ijk,i', x, self.A, x)
        # temp = np.mean(temp**2)
        temp = 1/2*np.einsum('j,ijk,i', x, self.A, x)
        var = np.var(temp)

        return -1/2 * x.T @ self.A_avg @ x + self.lmbda*var

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
        self.norm_gradF_traj = np.linalg.norm(
            self.grad_F_traj, ord=2, axis=0)**2
        self.norm_gradFdet_traj = np.linalg.norm(
            self.grad_Fdet_traj, ord=2, axis=0)**2

    def calculate_measure_avg(self):
        self.calculate_Dh()

        self.Dh1_avg_traj = np.cumsum(
            self.Dh1_traj)/np.arange(1, len(self.Dh1_traj)+1)
        self.Dh2_avg_traj = np.cumsum(
            self.Dh2_traj)/np.arange(1, len(self.Dh2_traj)+1)

        self.val_F_avg_traj = np.cumsum(
            self.val_F_traj) / np.arange(1, len(self.val_F_traj)+1)
        self.norm_gradFdet_avg_traj = np.cumsum(
            self.norm_gradFdet_traj) / np.arange(1, len(self.norm_gradFdet_traj)+1)
        self.norm_gradF_avg_traj = np.cumsum(
            self.norm_gradF_traj) / np.arange(1, len(self.norm_gradF_traj)+1)

    def plot(self, k1, k2, tau, avg=True):
        # if avg = True, plot the averaged D_h, if = False, plot each iteration

        fig, axs = plt.subplots(2, 3, figsize=(9, 8))
        fig.suptitle(f"k1={k1: .2e}, k2 = {k2: .2e}")

        if avg == True:
            if len(self.Dh1_avg_traj) == 0:
                self.calculate_measure_avg()
            axs[0, 0].plot(k1 * self.Dh1_avg_traj / tau**2)
            axs[0, 0].set_ylabel(r"$E[D_{h_1}(\hat{x}^{k+1}, x^k)/\tau^2]$")

            axs[0, 1].plot(k2 * self.Dh2_avg_traj / tau**2)
            axs[0, 1].set_ylabel(r"$E[D_{h_2}(\hat{x}^{k+1}, x^k)/\tau^2]$")

            axs[0, 2].plot((k1 * self.Dh1_avg_traj / tau**2 +
                           k2 * self.Dh2_avg_traj / tau**2))
            axs[0, 2].set_ylabel(r"$E[D_{h}(\hat{x}^{k+1}, x^k)/\tau^2]$")
            axs[1, 0].plot(self.val_F_avg_traj)
            axs[1, 0].set_ylabel(r"$E[F(x^k)]$")
            #axs[1, 0].set_ylim(1e0, 1e6)
            axs[1, 1].plot(self.norm_gradF_avg_traj)
            axs[1, 1].set_ylabel(r"$E[\|w^k\|^2]$")
            axs[1, 2].plot(self.norm_gradFdet_avg_traj)
            axs[1, 2].set_ylabel(r"$E[\|\nabla F(x^k)\|]^2$")
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
            axs[1, 1].plot(self.norm_gradF_traj)
            axs[1, 1].set_ylabel(r"$\|w^k\|^2$")
            axs[1, 2].plot(self.norm_gradFdet_traj)
            axs[1, 2].set_ylabel(r"$\|\nabla F(x^k)\|^2$")

        for ax in axs.flat:
            ax.set_xlabel("iteration")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.5)
        fig.tight_layout()
        axs[1, 0].set_yscale('linear')


class SCSC_SoR(Bregman_SoR):
    def __init__(self,  A, batch_size, x_init, alpha, beta, max_iter, R, lmbda, k1, k2, tau_Breg):
        # default setting for x_hat calculation
        self.alpha = alpha
        super().__init__(A, batch_size, x_init, k1, k2, tau_Breg, beta, max_iter, R, lmbda)

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
    def __init__(self,  A, batch_size, x_init, tau, beta, a, b, max_iter, R, lmbda, k1, k2, tau_Breg, beta_Breg):
        # default setting for x_hat calculation

        super().__init__(A, batch_size, x_init, k1, k2,
                         tau_Breg, beta_Breg, max_iter, R, lmbda)
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


class RoS:
    """class for RoR, RoS-VR algorithms, based on the policy evaluation problem, 

    Attributes:
        A: samples of A_xi
        batch_grad: batch size for inner/outer gradient, large batch of gradient for RoS-VR
        batch_grad_S: small batch of gradient for RoS-VR
        batch_val: batch size for inner function value, large batch for RoS-VR
        batch_val_S: small batch of inner value for RoS-VR
        max_iter: maximum iteration number of the algorithm
        max_k_iter, max_j_iter: outer and inner loop iteration number for RoS-VR
        x_init: initial x
        tau, lmbda: step size
        alpha, beta: coefs of SCSC
        beta_NASA, tau_NASA, a_NASA, b_NASA: coefs of NASA
        A_avg: E[A_xi]

    Result Attributes:
        x_traj: trajectory of x
        x_tilde_traj: trajectory of x_tilde
        Dh ||x_tilde^{k+1}- x^k||^2
        Dh_avg, Dh1_x_avg:1/k sum_0^k ||x_tilde^{k+1}- x^k||^2
        val_F_traj, val_F_avg_traj: trajectory of deterministic function value
        grad_Fdet_traj, grad_F_traj: trajectory of deterministic gradient and estimated gradient
        norm_gradFdet_traj, norm_gradFdet_avg_traj: (averaged) norm of deterministic gradient
        norm_gradF_traj, norm_gradF_avg_traj: (averaged) norm of estimated gradient
        oracle_grad, oracle_val: track of oracles up to each iteration

    """

    def __init__(self, A, b, tau, lmbda, alpha, beta, beta_NASA, tau_NASA, a_NASA, b_NASA, batch_grad, batch_grad_S, batch_val, batch_val_S, max_iter, max_k_iter, max_j_iter, x_init):
        self.A = A
        self.b = b
        self.n = A.shape[1]
        self.N = A.shape[2]
        self.A_avg = np.mean(A, 2)
        self.b_avg = np.mean(b, 1)
        self.Lf = self.b_avg.sum()
        self.tau = tau
        self.lmbda = lmbda
        self.alpha = alpha
        self.beta = beta
        self.beta_NASA = beta_NASA
        self.tau_NASA = tau_NASA
        self.a_NASA = a_NASA
        self.b_NASA = b_NASA
        self.max_iter = max_iter
        self.batch_grad_S = batch_grad_S
        self.batch_val_S = batch_val_S
        self.max_k_iter = max_k_iter
        self.max_j_iter = max_j_iter
        self.x_init = x_init

        self.batch_grad = batch_grad
        self.batch_val = batch_val

        self.x_traj = np.zeros([self.n, self.max_iter])  # traj of x
        # traj of x_tilde(calculated based on determinastic function)
        self.x_tilde_traj = np.zeros([self.n, self.max_iter])
        # traj of determinastic function value
        self.val_F_traj = np.zeros(self.max_iter)
        self.val_F_avg_traj = []
        # determinastic gradient and sample gradient
        self.prox_grad_traj = np.zeros([self.n, self.max_iter])
        self.grad_F_traj = np.zeros([self.n, self.max_iter])
        self.norm_gradFdet_traj = []
        self.norm_gradF_traj = []
        self.norm_gradFdet_avg_traj = []
        self.norm_gradF_avg_traj = []

        self.Dh_traj = np.zeros(self.max_iter)
        self.Dh_avg_traj = []

        self.oracle_grad = np.zeros(self.max_iter)
        self.oracle_val = np.zeros(self.max_iter)

    def _calculate_x_tilde(self, x_pre):
        g = self.A_avg @ x_pre

        grad_g = self.A_avg
        grad_f = 1 - self.b_avg/g

        grad_F = grad_g.T @ grad_f

        y = cp.Variable(self.n)
        constraints = [y >= 0]
        grad_hf = - 1/g
        hf_linear = - cp.sum(cp.log(g + grad_g@(y-x_pre)))

        sub_loss = (self.tau * grad_F - self.Lf * grad_g.T @ grad_hf) @ y + self.Lf * \
            hf_linear + self.lmbda / 2 * cp.sum_squares(y-x_pre)

        obj = cp.Minimize(sub_loss)
        prob = cp.Problem(obj, constraints)
        prob.solve()  # return the result of subproblem
        return y.value

    def _calculate_grad_F(self, x):
        g = self.A_avg @ x
        grad_g = self.A_avg
        grad_f = 1 - self.b_avg/g
        grad_F = grad_g.T @ grad_f
        return grad_F

    def _get_val_F(self, x):
        g = self.A_avg @ x
        val_F = np.sum(self.b_avg * np.log(self.b_avg) +
                       g - self.b_avg * np.log(g)-self.b_avg)
        return val_F

    def train_BG(self):
        x = self.x_init
        oracle_val = 0
        oracle_grad = 0

        self.x_traj[:, 0] = x
        self.x_tilde_traj[:, 0] = self._calculate_x_tilde(x)
        self.prox_grad_traj[:, 0] = self._calculate_grad_F(x)
        self.val_F_traj[0] = self._get_val_F(x)

        for iter in range(1, self.max_iter):
            # u update
            i = np.random.randint(0, self.N, self.batch_val)
            u = np.mean(self.A[:, :, i], 2) @ x
            # w update

            i = np.random.randint(0, self.N, self.batch_grad)
            v = np.mean(self.A[:, :, i], 2)

            i = np.random.randint(0, self.N, self.batch_grad)
            s = 1 - np.mean(self.b[:, i], 1)/u

            w = v.T @ s
            # solve the subproblem
            y = cp.Variable(self.n)
            constraints = [y >= 0]

            grad_hf = - 1/u
            hf_linear = - cp.sum(cp.log(u + v@(y-x)))

            sub_loss = (self.tau * w - self.Lf * v.T @ grad_hf) @ y + self.Lf * \
                hf_linear + self.lmbda / 2 * cp.sum_squares(y-x)

            obj = cp.Minimize(sub_loss)
            prob = cp.Problem(obj, constraints)
            prob.solve()  # return the result of subproblem

            # save the result
            x = y.value
            self.x_traj[:, iter] = x

            oracle_val = oracle_val + self.batch_val
            oracle_grad = oracle_grad + self.batch_grad
            self.oracle_grad[iter] = oracle_grad
            self.oracle_val[iter] = oracle_val

            self.grad_F_traj[:, iter] = w
            self.x_tilde_traj[:, iter] = self._calculate_x_tilde(x)
            self.prox_grad_traj[:, iter] = self._calculate_grad_F(x)
            self.val_F_traj[iter] = self._get_val_F(x)

    def train_BGVR(self):
        x = self.x_init
        oracle_val = 0
        oracle_grad = 0
        iter = 0

        self.x_traj[:, 0] = x
        self.x_tilde_traj[:, 0] = self._calculate_x_tilde(x)
        self.prox_grad_traj[:, 0] = self._calculate_grad_F(x)
        self.val_F_traj[0] = self._get_val_F(x)
        x_pre = x
        iter = 0
        for iter_k in range(self.max_k_iter):
            for iter_j in range(self.max_j_iter):
                iter = iter + 1
                if iter_j == 0:
                    # u update
                    i = np.random.randint(0, self.N, self.batch_val)
                    u = np.mean(self.A[:, :, i], 2) @ x

                    i = np.random.randint(0, self.N, self.batch_grad)
                    v = np.mean(self.A[:, :, i], 2)
                    oracle_val = oracle_val + self.batch_val
                    oracle_grad = oracle_grad + self.batch_grad
                else:
                    u_pre = u
                    v_pre = v
                    i = np.random.randint(0, self.N, self.batch_val_S)
                    u = u_pre + np.mean(self.A[:, :, i], 2) @ (x - x_pre)
                    v = v_pre
                    oracle_val = oracle_val + self.batch_val_S
                    oracle_grad = oracle_grad + self.batch_grad_S
                # s update
                i = np.random.randint(0, self.N, self.batch_grad)
                s = 1 - np.mean(self.b[:, i], 1) / u
                # solve the problem
                w = v.T @ s

                x_pre = x

                y = cp.Variable(self.n)
                constraints = [y >= 0]

                grad_hf = - 1/u
                hf_linear = - cp.sum(cp.log(u + v@(y-x)))

                sub_loss = (self.tau * w - self.Lf * v.T @ grad_hf) @ y + self.Lf * \
                    hf_linear + self.lmbda / 2 * cp.sum_squares(y-x)

                obj = cp.Minimize(sub_loss)
                prob = cp.Problem(obj, constraints)
                prob.solve()  # return the result of subproblem

                x = y.value
            # save the result
                self.x_traj[:, iter] = x

                self.oracle_grad[iter] = oracle_grad
                self.oracle_val[iter] = oracle_val

                self.grad_F_traj[:, iter] = w
                self.x_tilde_traj[:, iter] = self._calculate_x_tilde(x)
                self.prox_grad_traj[:, iter] = self._calculate_grad_F(x)
                self.val_F_traj[iter] = self._get_val_F(x)

    def train_NASA(self):
        oracle_val, oracle_grad = 0, 0
        x = self.x_init

        # initial sample
        i = np.random.randint(0, self.N, self.batch_val)
        u = np.mean(self.A[:, :, i], 2) @ x

        i = np.random.randint(0, self.N, self.batch_grad)
        v = np.mean(self.A[:, :, i], 2)

        i = np.random.randint(0, self.N, self.batch_grad)
        s = 1 - np.mean(self.b[:, i], 1)/u

        w = v.T @ s

        self.x_traj[:, 0] = x
        self.x_tilde_traj[:, 0] = self._calculate_x_tilde(x)
        self.prox_grad_traj[:, 0] = self._calculate_grad_F(x)
        self.val_F_traj[0] = self._get_val_F(x)
        self.grad_F_traj[:, 0] = w

        for iter in range(1, self.max_iter):
            x_pre = x
#            # update
            y = x - 1/self.beta_NASA * w
            y[y < 0] = 1e-16
            if iter == 1:
                x = y
            else:
                x = x_pre + self.tau_NASA * (y - x_pre)

            i = np.random.randint(0, self.N, self.batch_grad)
            s = 1 - np.mean(self.b[:, i], 1)/u
            i = np.random.randint(0, self.N, self.batch_grad)
            J = np.mean(self.A[:, :, i], 2)

            w = (1 - self.a_NASA * self.tau_NASA) * w + \
                self.a_NASA * self.tau_NASA * J.T @ s
            i = np.random.randint(0, self.N, self.batch_grad)
            g = np.mean(self.A[:, :, i], 2) @ x
            u = (1 - self.b_NASA * self.tau_NASA) * \
                u + self.b_NASA * self.tau_NASA * g
            self.x_traj[:, iter] = x

            oracle_val = oracle_val + self.batch_val
            oracle_grad = oracle_grad + self.batch_grad
            self.oracle_grad[iter] = oracle_grad
            self.oracle_val[iter] = oracle_val

            self.grad_F_traj[:, iter] = w
            self.x_tilde_traj[:, iter] = self._calculate_x_tilde(x)
            self.prox_grad_traj[:, iter] = self._calculate_grad_F(x)
            self.val_F_traj[iter] = self._get_val_F(x)

    def train_SCSC(self):

        oracle_val, oracle_grad = 0, 0
        x = self.x_init
        i = np.random.randint(0, self.N, self.batch_val)
        u = np.mean(self.A[:, :, i], 2) @ x

        i = np.random.randint(0, self.N, self.batch_grad)
        v = np.mean(self.A[:, :, i], 2)
        s = 1 - np.mean(self.b[:, i], 1) / u
        w = v.T @ s

        self.x_traj[:, 0] = x
        self.x_tilde_traj[:, 0] = self._calculate_x_tilde(x)
        self.prox_grad_traj[:, 0] = self._calculate_grad_F(x)
        self.val_F_traj[0] = self._get_val_F(x)
        self.grad_F_traj[:, 0] = w

        for iter in range(1, self.max_iter):
            x_pre = x

            x = x - self.alpha * w
            x[x < 0] = 1e-16

            i = np.random.randint(0, self.N, self.batch_val)
            g = np.mean(self.A[:, :, i], 2) @ x
            g_pre = np.mean(self.A[:, :, i], 2) @ x_pre

            u = (1 - self.beta) * (u + g - g_pre) + self.beta * g
            i = np.random.randint(0, self.N, self.batch_grad)
            v = np.mean(self.A[:, :, i], 2)
            i = np.random.randint(0, self.N, self.batch_grad)
            s = 1 - np.mean(self.b[:, i], 1) / u

            w = v.T @ s

            oracle_val = oracle_val + self.batch_val
            oracle_grad = oracle_grad + self.batch_grad
            self.x_traj[:, iter] = x

            self.oracle_grad[iter] = oracle_grad
            self.oracle_val[iter] = oracle_val

            self.grad_F_traj[:, iter] = w
            self.x_tilde_traj[:, iter] = self._calculate_x_tilde(x)
            self.prox_grad_traj[:, iter] = self._calculate_grad_F(x)
            self.val_F_traj[iter] = self._get_val_F(x)

    def calculate_measure(self):
        # calculate Dh for x_tilde
        self.Dh_traj = np.linalg.norm(
            self.x_tilde_traj - self.x_traj, ord=2, axis=0) ** 2
        self.norm_gradF_traj = np.linalg.norm(
            self.grad_F_traj, ord=2, axis=0) ** 2
        self.norm_gradFdet_traj = np.linalg.norm(
            self.prox_grad_traj, ord=2, axis=0) ** 2

        self.norm_gradFdet_avg_traj = []
        self.norm_gradF_avg_traj = []
        # average
        self.Dh_avg_traj = np.cumsum(
            self.Dh_traj) / np.arange(1, len(self.Dh_traj)+1)
        self.norm_gradF_avg_traj = np.cumsum(
            self.norm_gradF_traj) / np.arange(1, len(self.norm_gradF_traj)+1)
        self.norm_gradFdet_avg_traj = np.cumsum(
            self.norm_gradFdet_traj) / np.arange(1, len(self.norm_gradFdet_traj)+1)
        self.val_F_avg_traj = np.cumsum(
            self.val_F_traj) / np.arange(1, len(self.val_F_traj)+1)

    def plot(self, tau, lmbda, avg=True):
        # if avg = True, plot the averaged D_h, if = False, plot each iteration

        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle(f"tau={tau: .2e}, lambda = {lmbda: .2e}")

        if avg == True:
            axs[0, 0].plot(self.Dh_avg_traj / tau**2)
            axs[0, 0].set_ylabel(r"$E[D_{h}(\tilde{x}^{k+1}, x^k)/\tau^2]$")
            axs[0, 1].plot(self.val_F_avg_traj)
            axs[0, 1].set_ylabel(r"$E[F(x^k)]$")
            axs[1, 0].plot(self.norm_gradF_avg_traj[1:-1])
            axs[1, 0].set_ylabel(r"$E[\|w^k\|^2]$")
            axs[1, 1].plot(self.norm_gradFdet_avg_traj)
            axs[1, 1].set_ylabel(r"$E[\|\nabla F(x^k)\|]^2$")
        else:
            axs[0, 0].plot(self.Dh_traj / tau**2)
            axs[0, 0].set_ylabel(r"$E[D_{h}(\tilde{x}^{k+1}, x^k)/\tau^2]$")
            axs[0, 1].plot(self.val_F_traj)
            axs[0, 1].set_ylabel(r"$E[F(x^k)]$")
            axs[1, 0].plot(self.norm_gradF_traj[1:-1])
            axs[1, 0].set_ylabel(r"$E[\|w^k\|^2]$")
            axs[1, 1].plot(self.norm_gradFdet_traj)
            axs[1, 1].set_ylabel(r"$E[\|\bar{x}^{k+1} - x^k\|]^2/\tau^2$")
        for ax in axs.flat:
            ax.set_xlabel("iteration")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.5)
        fig.tight_layout()
