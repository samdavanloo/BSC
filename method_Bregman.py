import numpy as np
import cvxpy as cp  # solver used for the subproblem


def SoR_breg(A, lmbda, R, tau_k, beta_k, n_batch, c1, c2, iter_max, x_init, track_f=1):

    d = A.shape[0]
    n = A.shape[2]

    w = np.random.randn(d)
    u = np.random.randn(2)

    A_avg = A.mean(axis=2)
    A_norm = np.linalg.norm(A_avg, 2)

    f_BG = np.zeros(iter_max)
    x_histroy = np.zeros([d, iter_max])
    x_hat_history = np.zeros([d, iter_max])
    x_BG = np.zeros([d, iter_max])

    x = x_init

    for iter in range(iter_max):
        grad_h = c2 * np.linalg.norm(x)**2 * x + c1*x
        c = w*tau_k - grad_h

        if np.all(c == 0):
            y = c
        else:
            p = [c2*np.linalg.norm(c)**2, 0, c1,  -1]
            theta = np.roots(p)
            theta = theta[np.isreal(theta)]
            theta = np.real(theta.max())
            y = -theta*c
            y = y * min(1, R/np.linalg.norm(y))
        x_pre = x
        x = y

        if n_batch > 1:
            i = np.random.randint(0, n, n_batch)

            temp1 = 1/2 * x_pre.T @ np.mean(A[:, :, i], 2) @ x_pre
            temp2 = 1/2 * np.einsum('j,ijk,i', x_pre, A[:, :, i], x_pre)
            temp2 = np.mean(temp2**2)
            G_pre = np.array([temp1, temp2])

            temp1 = 1/2 * x.T @ np.mean(A[:, :, i], 2) @ x
            temp2 = 1/2 * np.einsum('j,ijk,i', x, A[:, :, i], x)
            temp2 = np.mean(temp2**2)
            G = np.array([temp1, temp2])
            u = (1-beta_k)*(u+G - G_pre) + beta_k*G

            i = np.random.randint(0, n, n_batch)
            # calculate each sample
            J_1 = np.einsum('ijk,j->ik', A[:, :, i], x)

            J_21 = np.einsum('j,ijk,i', x, A[:, :, i], x)
            J_2 = J_21*J_1

            J = np.row_stack([J_1.mean(axis=1), J_2.mean(axis=1)])
        elif n_batch == 1:
            i = np.random.randint(n)
            temp1 = 1/2 * x_pre.T @ A[:, :, i] @ x_pre
            G_pre = np.array([temp1, temp1**2])

            temp1 = 1/2 * x.T @ A[:, :, i] @ x
            G = np.array([temp1, temp1**2])
            u = (1-beta_k)*(u+G - G_pre) + beta_k*G

            i = np.random.randint(n)
            J = np.row_stack(
                [A[:, :, i]@x, (x.T @ A[:, :, i] @ x) * A[:, :, i] @ x])

        s = np.array([-1-2*lmbda*u[0], lmbda])
        w = J.T@s
        x_histroy[:, iter] = x

        xk = np.mean(x_histroy[:, 0:iter+1], 1)
        x_BG[:, iter] = xk

        # temp = 0
        # for k in range(n):
        #     temp = temp + 1/4*(xk.T@A[:, :, k]@xk)**2
        # temp = temp/n
        if track_f == 1:

            temp = np.einsum('j,ijk,i', xk, A, xk)
            temp = np.mean(temp**2)
            f_BG[iter] = -1/2*xk.T@A_avg@xk + \
                lmbda*(temp - 1/4*(xk.T@A_avg@xk)**2)
        # grad_norm[iter] = np.linalg.norm(w)
        
        
        #############  Calculate x_hat #####################
        g1 = 1/2 * x_pre.T @ A_avg @ x_pre
        g2 = 1/2 * np.einsum('j,ijk,i', x_pre, A, x_pre)
        g2 = np.mean(g2**2)
        g_true = np.array([g1, g2])
        grad_f = np.array([-1-2*lmbda*g_true[0], lmbda])
        
        J_1 = np.einsum('ijk,j->ik', A, x_pre)
        J_21 = np.einsum('j,ijk,i', x_pre, A, x_pre)
        J_2 = J_21*J_1

        grad_g = np.row_stack([J_1.mean(axis=1), J_2.mean(axis=1)])
        
        grad_F = grad_g.T @ grad_f
        
        
        ### solve subproblem based on true gradient
        c = grad_F*tau_k/2 - grad_h

        if np.all(c == 0):
            y = c
        else:
            p = [c2*np.linalg.norm(c)**2, 0, c1,  -1]
            theta = np.roots(p)
            theta = theta[np.isreal(theta)]
            theta = np.real(theta.max())
            y = -theta*c
            y = y * min(1, R/np.linalg.norm(y))
        
        x_hat_history[:,iter] = y
        
    return f_BG, x_BG, x_histroy, x_hat_history


def RoS_breg(A, b, tau, lmbda, batch_grad, batch_val, iter_max, x_init):
    m = A.shape[0]
    n = A.shape[1]
    N = A.shape[2]

    A_avg = np.mean(A, 2)
    b_avg = np.mean(b, 1)

    Lf = b_avg.sum()

    x_history = np.zeros([n, iter_max])
    #x_out = np.zeros([n, iter_max])
    x_hat_history = np.zeros([n, iter_max])
    oracle_innervalue = np.zeros(iter_max)
    oracle_innergrad = np.zeros(iter_max)
    x = x_init
    oracle_val = 0
    oracle_grad = 0
    for iter in range(iter_max):
        x_pre = x
        # u update
        if batch_val > 1:

            i = np.random.randint(0, N, batch_val)
            u = np.mean(A[:,:,i],2) @ x
            
            
        else:
            i = np.random.randint(N)
            u = A[:, :, i]@x

        # w update
        if batch_grad > 1:
            i = np.random.randint(0, N, batch_grad)
            v = np.mean(A[:, :, i], 2)

            i = np.random.randint(0, N, batch_grad)
            s = 1 - np.mean(b[:, i],1)/u

            w = v.T @ s
        else:
            i = np.random.randint(N)
            v = A[:, :, i]

            i = np.random.randint(N)
            s = 1 - b[:, i]/u

            w = v.T @ s
        # solve the subproblem
        y = cp.Variable(n)
        constraints = [y >= 0]

        grad_hf = - 1/u
        hf_linear = - cp.sum(cp.log(u + v@(y-x)))

        sub_loss = (tau * w - Lf * v.T @ grad_hf) @ y + Lf * \
            hf_linear + lmbda / 2 * cp.sum_squares(y-x)

        obj = cp.Minimize(sub_loss)
        prob = cp.Problem(obj, constraints)
        prob.solve()  # return the result of subproblem

        # save the result
        x = y.value

        x_history[:, iter] = x
        
        oracle_val = oracle_val +batch_val
        oracle_grad = oracle_grad + batch_grad
        
        oracle_innervalue[iter] = oracle_val
        oracle_innergrad[iter] = oracle_grad
        
       # x_out[:, iter] = np.mean(x_history[:, 0:iter+1], 1)

       # f[iter] = np.sum(
       #     b_avg * np.log(b_avg/(A_avg @ x_out[:, iter])) + A_avg @ x_out[:, iter] - b_avg)

        # Calculate x_tilde
        
        g = A_avg @ x_pre
       
        grad_g = A_avg
        grad_f = 1 - np.mean(b,axis=1)/g
        
        grad_F = grad_g.T @ grad_f
       
        
        y = cp.Variable(n)
        constraints = [y >= 0]

        grad_hf = - 1/g
        hf_linear = - cp.sum(cp.log(g + grad_g@(y-x_pre)))

        sub_loss = (tau * grad_F - Lf * grad_g.T @ grad_hf) @ y + Lf * \
            hf_linear + lmbda / 2 * cp.sum_squares(y-x_pre)

        obj = cp.Minimize(sub_loss)
        prob = cp.Problem(obj, constraints)
        prob.solve()  # return the result of subproblem
        
        x_tilde = y.value
        x_hat_history[:,iter] = x_tilde
    return x_history,x_hat_history,oracle_innervalue,oracle_innergrad





def RoS_VR(A, b, tau, lmbda, batch_grad_B,batch_grad_S,
               batch_val_B, batch_val_S, iter_j_max, iter_k_max, x_init):
    m = A.shape[0]
    n = A.shape[1]
    N = A.shape[2]

    A_avg = np.mean(A, 2)
    b_avg = np.mean(b, 1)

    Lf = b_avg.sum()

    x_history = np.zeros([n, iter_j_max*iter_k_max])
    #x_out = np.zeros([n, iter_max])
    x_tilde_history = np.zeros([n, iter_j_max*iter_k_max])
    oracle_innervalue = np.zeros(iter_j_max*iter_k_max)
    oracle_innergrad = np.zeros(iter_j_max*iter_k_max)

    x = x_init
    oracle = 0
    oracle_grad = 0
    oracle_idx = 0
   # x_pre = x
    for iter_k in range(iter_k_max):
      
        # small batch size (j > 0)
        for iter_j in range(iter_j_max):
            if iter_j == 0:
            # large batch sieze (j = 0)
                # u_0
                if batch_val_B >1 :
                    i = np.random.randint(0,N,batch_val_B)
                    u = np.mean(A[:,:,i],2) @ x
                else:
                    i = np.random.randint(N)
                    u = A[:, :, i]@x
                # v update
                if batch_grad_B > 1:
                    i = np.random.randint(0, N, batch_grad_B)
                    v = np.mean(A[:, :, i], 2)       
                else:
                    i = np.random.randint(N)
                    v = A[:, :, i]
                oracle = oracle+batch_val_B
                oracle_grad = oracle_grad + batch_grad_B
            else:
                u_pre = u
                v_pre = v
                # u update
                if batch_val_S >1:
                    i = np.random.randint(0,N,batch_val_S)
                    u = u_pre  + np.mean(A[:,:,i],2) @ (x - x_pre)
                else:
                    i = np.random.randint(0,N,batch_val_S)
                    u = u_pre  + A[:,:,i] @ (x - x_pre)
                    
                # v update (grad g_xi = A_xi, hence the gradient for two points are same)
                    v = v_pre 
                oracle = oracle+batch_val_S
                oracle_grad = oracle_grad + batch_grad_S
            # s update
            if batch_grad_B >1 :
                i = np.random.randint(0,N,batch_grad_B)
                s = 1-np.mean(b[:,i],1)/u
            else:
                i = np.random.randint(0,N,batch_grad_B)
                s = 1-b[:,i]/u
                
            # solve the problem
            w = v.T @ s
            
            x_pre = x
            
            y = cp.Variable(n)
            constraints = [y >= 0]
    
            grad_hf = - 1/u
            hf_linear = - cp.sum(cp.log(u + v@(y-x)))
    
            sub_loss = (tau * w - Lf * v.T @ grad_hf) @ y + Lf * \
                hf_linear + lmbda / 2 * cp.sum_squares(y-x)
    
            obj = cp.Minimize(sub_loss)
            prob = cp.Problem(obj, constraints)
            prob.solve()  # return the result of subproblem
    
            # save the result
            x = y.value
            
            x_history[:,iter_k*iter_j_max+iter_j] = x
            
            oracle_innervalue[oracle_idx] = oracle
            oracle_innergrad[oracle_idx] = oracle_grad
            
            oracle_idx +=1
            
            # Calculate x_tilde
        
            g = A_avg @ x_pre
           
            grad_g = A_avg
            grad_f = 1 - np.mean(b,axis=1)/g
            
            grad_F = grad_g.T @ grad_f
           
            
            y = cp.Variable(n)
            constraints = [y >= 0]
    
            grad_hf = - 1/g
            hf_linear = - cp.sum(cp.log(g + grad_g@(y-x_pre)))
    
            sub_loss = (tau * grad_F - Lf * grad_g.T @ grad_hf) @ y + Lf * \
                hf_linear + lmbda / 2 * cp.sum_squares(y-x_pre)
    
            obj = cp.Minimize(sub_loss)
            prob = cp.Problem(obj, constraints)
            prob.solve()  # return the result of subproblem
            
            x_tilde = y.value
            x_tilde_history[:,iter_k*iter_j_max+iter_j] = x_tilde
            
    return x_history,x_tilde_history, oracle_innervalue, oracle_innergrad

        
        
