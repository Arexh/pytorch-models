import numpy as np
from scipy.optimize import minimize
from scipy.optimize import nnls

class PELTR():
    def __init__(self):
        self.w1 = 0.5
        self.w2 = 0.5
        self.c1 = 0.6
        self.c2 = 0.2


    def solve(self, G1, G2):
        G1 = np.expand_dims(np.transpose(G1.cpu()), axis=0)
        G2 = np.expand_dims(np.transpose(G2.cpu()), axis=0)
        G = np.vstack((G1, G2))
        self.w1, self.w2 = pareto_step(np.mat([[self.w1], [self.w2]]), np.mat([[self.c1], [self.c2]]), G)
        return self.w1, self.w2


def pareto_step(w, c, G):
    """
    ref:http://ofey.me/papers/Pareto.pdf
    K : the number of task
    M : the dim of NN's params
    :param W: # (K,1)
    :param C: # (K,1)
    :param G: # (K,M)
    :return:
    """
    GGT = np.matmul(G, np.transpose(G))  # (K, K)
    e = np.mat(np.ones(np.shape(w)))  # (K, 1)
    m_up = np.hstack((GGT, e))  # (K, K+1)
    m_down = np.hstack((np.transpose(e), np.mat(np.zeros((1, 1)))))  # (1, K+1)
    M = np.vstack((m_up, m_down))  # (K+1, K+1)
    z = np.vstack((-np.matmul(GGT, c), 1 - np.sum(c)))  # (K+1, 1)
    hat_w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M)), M), z)  # (K+1, 1)
    hat_w = hat_w[:-1]  # (K, 1)
    hat_w = np.reshape(np.array(hat_w), (hat_w.shape[0],))  # (K,)
    c = np.reshape(np.array(c), (c.shape[0],))  # (K,)
    new_w = ASM(hat_w, c)
    return new_w


def ASM(hat_w, c):
    """
    ref:
    http://ofey.me/papers/Pareto.pdf,
    https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1
    :param hat_w: # (K,)
    :param c: # (K,)
    :return:
    """
    A = np.array([[0 if i != j else 1 for i in range(len(c))] for j in range(len(c))])
    b = hat_w
    x0, _ = nnls(A, b)

    def _fn(x, A, b):
        return np.linalg.norm(A.dot(x) - b)

    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) + np.sum(c) - 1}
    bounds = [[0., None] for _ in range(len(hat_w))]
    min_out = minimize(_fn, x0, args=(A, b), method='SLSQP', bounds=bounds, constraints=cons)
    new_w = min_out.x + c
    return new_w
