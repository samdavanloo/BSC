import numpy as np
import matplotlib.pyplot as plt


class Bregman_SoR():
    def __init__(self,  A, batch_size, x_init, k1, k2, tau, beta, max_iter, R, lmbda=1):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.A = A
        self.x_init = x_init  # initial x
        self.k1 = k1
        self.k2 = k2
        self.tau = tau
        self.beta = beta
        self.lmbda = lmbda
        self.R = R  # radius of feasible set

        self.d = A.shape[0]
        self.n = A.shape[2]  # number of total samples
        self.A_avg = self.A.mean(axis=2)

        self.x_traj = np.zeros([self.d, self.max_iter])  # traj of x
        # traj of x_hat(calculated based on determinastic function)
        self.x_hat_traj = np.zeros([self.d, self.max_iter])
        # traj of determinastic function value
        self.val_F_traj = np.zeros(self.max_iter)
        # D_h1(x_hat^{k+1}- x_hat^k) with tau and k1
        self.Dh1_x_hat = []
        # D_h2(x_hat^{k+1}- x_hat^k) with tau and k2
        self.Dh2_x_hat = []

    def __get_val_g(self, A_sample, x):
        # get sample value of inner function g
        g1 = 1/2 * x.T @ np.mean(A_sample, 2) @ x
        g2 = 1/2 * np.einsum('j, ijk, i', x, A_sample, x)
        g2 = np.mean(g2**2)

        return np.array([g1, g2])  # g value

    def __get_val_F(self, x):
        temp = np.einsum('j,ijk,i', x, self.A, x)
        temp = np.mean(temp**2)

        return -1/2 * x.T @ self.A_avg @ x + self.lmbda*(temp - 1/4*(x.T @ self.A_avg @ x)**2)

    def __get_grad_h(self, x):
        # get gradient of geneartining function h(x)
        return self.k1 * x + self.k2 * np.linalg.norm(x)**2 * x

    def __get_grad_g(self, A_sample, x):
        # get sample gradient of inner function g
        grad_g1 = np.einsum('ijk,j->ik', A_sample, x)
        grad_g21 = np.einsum('j,ijk,i', x, A_sample, x)
        grad_g2 = grad_g21 * grad_g1

        return np.row_stack([grad_g1.mean(axis=1), grad_g2.mean(axis=1)])

    def __get_grad_f(self, u):
        # get gradient of outter function f(which is deterministic )
        return np.array([-1 - 2 * self.lmbda * u[0], self.lmbda])

    def __solve_Breg_sub(self, x, w):
        # solve the Bregman subproblem
        grad_h = self.__get_grad_h(x)
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

    def __update_u(self, A_sample, u, x_pre, x):
        g_pre = self.__get_val_g(A_sample, x_pre)
        g = self.__get_val_g(A_sample, x)

        u = (1 - self.beta) * (u + g - g_pre) + self.beta * g

        return u

    def __sample_A(self):
        idx_sample = np.random.randint(0, self.n, self.batch_size)
        return self.A[:, :, idx_sample]

    def train(self):
        x = self.x_init

        # initial sample
        A_sample = self.__sample_A()
        u = self.__get_val_g(A_sample, x)
        v = self.__get_grad_g(A_sample, x)
        s = self.__get_grad_f(u)
        w = v.T @ s

        for iter in range(self.max_iter):
            x_pre = x
            x = self.__solve_Breg_sub(x, w)

            A_sample = self.__sample_A()
            u = self.__update_u(A_sample, u, x_pre, x)
            v = self.__get_grad_g(A_sample, x)
            s = self.__get_grad_f(u)
            w = v.T @ s

            self.x_traj[:, iter] = x

            # calculate x_hat
            val_g = self.__get_val_g(self.A, x)
            v_det = self.__get_grad_g(self.A, x)
            s_det = self.__get_grad_f(val_g)
            grad_F = v_det.T @ s_det
            x_hat = self.__solve_Breg_sub(x, grad_F)
            self.x_hat_traj[:, iter] = x_hat

            self.val_F_traj[iter] = self.__get_val_F(x)

    def plot(self):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "ps.usedistiller": "xpdf"})

        self.Dh1_x_hat = self.x_hat_traj[:, 1:] - self.x_traj[:, 0:-1]
        # self.Dh1_x_hat = self.k1 / \
        #     (self.tau**2) * np.sum(self.Dh1_x_hat * self.Dh1_x_hat, axis=0)
        self.Dh1_x_hat =  np.sum(self.Dh1_x_hat * self.Dh1_x_hat, axis=0)

        self.Dh2_x_hat = np.zeros(self.max_iter - 1)

        for iter in range(0, self.max_iter - 1):
            y = self.x_hat_traj[:, iter+1]
            x = self.x_traj[:, iter]
            # self.Dh2_x_hat[iter] = self.k2 / (self.tau**2) * (np.linalg.norm(
            #     y)**4 / 4 - np.linalg.norm(x)**4 / 4 - np.linalg.norm(x)**2 * x @ (y-x))
            self.Dh2_x_hat[iter] = (np.linalg.norm(
                y)**4 / 4 - np.linalg.norm(x)**4 / 4 - np.linalg.norm(x)**2 * x @ (y-x))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize =(6,4))
        fig.suptitle(f"k1={self.k1: .2f}, k2 = {self.k2: .2f}")
        ax1.plot(self.Dh1_x_hat)
        ax1.set_ylabel(r"$D_{h_1}(\hat{x}^{k+1}-x^k)/\tau^2$")

        ax2.plot(self.Dh2_x_hat)
        ax2.set_ylabel(r"$D_{h_2}(\hat{x}^{k+1}-x^k)/\tau^2$")

        for ax in fig.get_axes():
            ax.set_xlabel("iteration")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.5)
        fig.tight_layout()

        # fig.show()


if __name__ == '__main__':
    pass
