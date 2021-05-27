from functions import *


def solution(y, p, x_0=1, tau=1.01, max_k=200):
    solutions = []
    conv_ens = []
    A = regression_matrix(y, p)
    u, s, vt = np.linalg.svd(A)
    alpha = ((choose_alpha(A, y, u, s, vt, p))) / len(y)

    if x_0 == 1:
        sign = get_sign_vector(A, y, p, alpha)
        s_filtered = np.vstack([np.diag(filter_k(s, alpha, 1)), np.zeros([u.shape[1] - len(s), len(s)])])
        s_filtered_x0 = np.diag(filter_k_x0(s, alpha, 1))
        x_0 = vt.T @ s_filtered.T @ u.T @ y[p:] - 1 * vt.T @ s_filtered_x0 @ vt @ sign
    else:
        x_0 = np.zeros(p)

    delta_est = delta_tsvd(A, y, u, s, vt, p)

    for i in range(1, max_k):

        s_filtered = np.vstack([np.diag(filter_k(s, alpha, i)), np.zeros([u.shape[1] - len(s), len(s)])])
        s_filtered_x0 = np.diag(filter_k_x0(s, alpha, i))

        x_i = vt.T @ s_filtered.T @ u.T @ y[p:] + vt.T @ s_filtered_x0 @ vt @ x_0
        solutions.append(x_i)
        conv_ens.append(np.linalg.norm(A @ x_i - y[p:]) * np.linalg.norm(x_i))

        if np.linalg.norm(A @ x_i - y[p:]) <= tau * delta_est:
            return x_i

    i_argmin = np.argmin(conv_ens)

    return solutions[i_argmin]

print(solution([1,2,3,4,5,6,7], 2))