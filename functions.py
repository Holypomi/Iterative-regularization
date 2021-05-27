import numpy as np
from sklearn.linear_model import ElasticNetCV


def regression_matrix(y, p):
    n = len(y)
    columns = []
    for i in range(0, p):
        columns.append(y[i:n - p + i])
    return np.dstack(columns[::-1])[0]


def delta_tsvd(A, y, u, s, vt, p):
    values = []
    for i in range(len(s) + 1):
        s_temp = np.concatenate([s[:i], np.zeros(vt.shape[0] - i)])
        s_temp_filtered = np.vstack([np.diag(filter_k(s_temp, 0, 1)), np.zeros([u.shape[1] - len(s), len(s)])])
        C = vt.T @ s_temp_filtered.T @ u.T
        mat = np.ones(s_temp_filtered.shape[0]) - A @ C
        values.append(((np.linalg.norm(mat @ y[p:]) ** 2) / (np.trace(mat) ** 2), C))
    C_opt = sorted(values, key=lambda x: x[0])[0][1]
    delta_est = np.linalg.norm(A @ C_opt - y[p:])

    return delta_est


def get_sign_vector(A, y, p, alpha):
    alpha_2 = alpha
    alpha_1_params = np.array([10 ** k for k in range(-6, 6, 1)])
    l1_ratios = alpha_1_params / (alpha_1_params + alpha_2)
    model = ElasticNetCV(l1_ratio=l1_ratios, alphas=alpha_1_params)
    sol = model.fit(A, y[p:]).coef_
    return np.sign(sol)


def choose_alpha(A, y, u, s, vt, p):
    lambda_1 = max(np.linalg.eigvals(A.T @ A))

    def functional_value(alpha):

        s_filtered_alpha = np.vstack([np.diag(filter_k(s, alpha, 1)), np.zeros([u.shape[1] - len(s), len(s)])])
        C_alpha = vt.T @ s_filtered_alpha.T @ u.T
        x_hat = C_alpha @ y[p:]
        X = A @ C_alpha

        v_sq = (np.linalg.norm(A @ x_hat - y[p:]) ** 2) / len(y[p:])
        y_delta_sq = np.linalg.norm(y[p:]) ** 2 - np.linalg.norm(A @ x_hat - y[p:]) ** 2
        frob_norm = np.trace(X.T @ X)
        beta_min = (alpha / (lambda_1 + alpha)) ** 2

        return beta_min * y_delta_sq + v_sq * frob_norm

    minn = np.infty
    new_alpha = 0
    for alpha in [10**i for i in range(-6, 7, 1)]:
        t = functional_value(alpha)
        if t < minn:
            minn = t
            new_alpha = alpha

    return new_alpha


def filter_k(s, alpha, k):
    t = s.copy()
    for i in range(len(t)):
        if t[i] != 0:
            t[i] = (1 / t[i]) * (1 - (alpha ** 2 / (t[i] ** 2 + alpha ** 2)) ** k)
    return t


def filter_k_x0(s, alpha, k):
    t = s.copy()
    for i in range(len(t)):
        if t[i] != 0:
            t[i] = (alpha ** 2 / (t[i] ** 2 + alpha ** 2)) ** k
    return t