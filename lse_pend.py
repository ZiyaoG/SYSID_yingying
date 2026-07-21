import scipy as sc
import numpy as np


def theta_star_values(m: float, l: float) -> np.ndarray:
    """Ground-truth regression parameters [1/l, 1/(m l^2)]."""
    return np.array([1.0 / l, 1.0 / (m * l * l)], dtype=float)


def error_theta_l2_rel(theta_hat: np.ndarray, theta_star_vec: np.ndarray) -> float:
    """||theta_hat - theta_star||_2 / ||theta_star||_2."""
    return float(
        np.linalg.norm(theta_hat - theta_star_vec)
        / (np.linalg.norm(theta_star_vec) + 1e-12)
    )


def run_lst(S, Phi_S_U):

    Y: object = []
    X: object = []

    for t in range(len(Phi_S_U)):

        delta_S = S[t]
        phi_s_u = Phi_S_U[t]

        Y.append([delta_S])
        X.append([phi_s_u[0], phi_s_u[1]])


    YY = np.array(Y)
    XX = np.array(X)

    theta_hat = (sc.linalg.pinv(XX) @ YY).T
    theta_hat_ = np.array([theta_hat[0,0], theta_hat[0,1]])
    return theta_hat_
