import numpy as np
from scipy.stats import truncnorm
import random
import matplotlib.pyplot as plt
import math

g = 9.81  # (m/s^2)  gravity constant
dt = 0.01  # time_step for discrete-time system

def system_parameters():
    m = 0.1  # (kg)    mass
    l = 0.5  # (m) distance between the rotor and the center of mass
    k = 0.1  # controller gain, LSE uses k=2 and SME uses k=0.1
    return m, l, k

def generate_u(input_, time_hor, s_, mean, std, u_max, lb, ub):  # noise in control input
    if input_ == "trunc_guass":
        np.random.seed(s_)
        rv = truncnorm(-u_max, u_max, loc=mean, scale=std)
        r1 = rv.rvs(size=time_hor)
        return r1
    elif input_ == "uniform":
        np.random.seed(s_)
        r1 = np.random.uniform(low=lb, high=ub, size=time_hor)
        return r1
    elif input_ == "bernouli":
        np.random.seed(s_)
        r0 = np.random.rand(time_hor)
        r1 = []
        for k in range(len(r0)):
          if r0[k] < 0.5:
            r1.append(0.5)
          else:
            r1.append(-0.5)
        return r1

def generate_w(distr, time_hor, s_, mean, std, w_max, lb, ub):  # disturbance
    if distr == "trunc_guass":
        np.random.seed(s_)
        rv = truncnorm(-w_max, w_max, loc=mean, scale=std)
        r1 = rv.rvs(size=time_hor)
        np.random.seed(s_+1000)
        rv = truncnorm(-w_max, w_max, loc=mean, scale=std)
        r2 = rv.rvs(size=time_hor)
        return r1, r2
    elif distr == "uniform":
        np.random.seed(s_)
        r1 = np.random.uniform(low=lb, high=ub, size=time_hor)
        np.random.seed(s_+1000)
        r2 = np.random.uniform(low=lb, high=ub, size=time_hor)
        return r1, r2
    elif distr == "bernouli":
        np.random.seed(s_)
        r01 = np.random.rand(time_hor)
        np.random.seed(s_+1000)
        r02 = np.random.rand(time_hor)
        r1 = []
        r2 = []
        for k in range(len(r01)):
          if r01[k] < 0.5:
            r1.append(100)
          else:
            r1.append(-100)
        for k in range(len(r02)):
          if r02[k] < 0.5:
            r2.append(100)
          else:
            r2.append(-100)
        return r1, r2

class SimplePendulumDynamics:
    def __init__(self, distr, input):
        self.state = None
        self.u0 = None
        self.distr = distr
        self.input = input
        self.m, self.l, self.k = system_parameters()
        self.alpha_list = []
        self.omega_list = []
        self.phi_s_u_list = []
        self.b_s_list = []
        self.phi_list = []
        self.state_list = []

    def plot_trajectory(self):
        t_list = np.array(range(len(self.alpha_list))) * dt
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.4)   # wspace: space between subplots in a row

        # print(np.sin(np.array(self.alpha_list)))
        axs[0].plot(t_list, np.sin(np.array(self.alpha_list)), label='$sin(\\alpha)$')
        axs[0].plot(t_list, np.array(self.alpha_list), label='$\\alpha$')
        # axs[0].scatter(t_list, np.sin(np.array(self.alpha_list)), marker='o')
        # plt.title("Quadrotor's Position")
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('$\\alpha$ ($rad$)')
        axs[0].legend()

        # print(np.array(self.omega_list))
        axs[1].plot(t_list, self.omega_list, label='$\omega$')
        # axs[1].scatter(t_list, self.omega_list, marker='o')
        # plt.title("Quadrotor's Angular Velocity")
        axs[1].set_xlabel('time (s)')
        axs[1].set_ylabel('$\omega$ ($rad/s^{2}$)')
        axs[1].legend()

        plt.show()

    def update_feature_list(self, phi_s_u, s_, s, ex):
        self.phi_s_u_list.append(phi_s_u)
        self.b_s_list.append(s - s_ - ex)

    def update_feat(self, y):
        self.phi_list.append(y)


    def get_trajectory_3(
        self,
        x0,
        time_hor,
        s_u,
        s_w,
        param_u,
        mult_u,
        param_w,
        mult_w,
        u_bound=None,
    ):

        # ----------------------------------------- initial states -----------------------------------------------------
        self.state = x0
        x = np.array(x0)
        alpha_ = x[0]  # angle
        omega_ = x[1]  # angular velocity

        #  ------------------------------------- Storing the states - ---------------------------------------------
        self.alpha_list = [alpha_]
        self.omega_list = [omega_]
        self.state_list = [np.array([alpha_, omega_])]

        if self.input == "trunc_guass":
          u_max_ = param_u[2]
        else:
          u_max_ = 1.0

        if self.distr == "trunc_guass":
          w_max_ = param_w[2]
        else:
          w_max_ = 1.0

        # -----------------  random noise and disturbance generation ---------------------------------------------------
        U1_list = generate_u(self.input, time_hor, s_u, mean=param_u[0], std=param_u[1], u_max=u_max_, lb=param_u[0], ub=param_u[1])
        W1_list, W2_list = generate_w(self.distr, time_hor, s_w, mean=param_w[0], std=param_w[1], w_max=w_max_, lb=param_w[0], ub=param_w[1])

        for t in range(time_hor):

            s_ = omega_

            # ------------------  noise in control input  (for exploration)  ----------------------------------------
            u1 = U1_list[t]

            # ----------------   noise in control input  (for exploration)  -----------------------------------------
            w1 = mult_w[0] * W1_list[t]
            w2 = mult_w[1] * W2_list[t]

            # ----------------------------------------  PD control + noise  ------------------------------------------
            u = - self.k * omega_ + u1
            if u_bound is not None:
                u = float(np.clip(u, -u_bound, u_bound))

            # ------------------------------------------  Dynamic model ----------------------------------------------
            alpha_dot = omega_ + w1
            omega_dot = - g * math.sin(alpha_) / self.l + u / (self.m * self.l * self.l) + w2

            phi_s_u = np.array([-g*math.sin(alpha_), u])
            self.update_feat(phi_s_u)

            # -------------------------------------- Updating the states --------------------------------------------
            alpha = alpha_ + dt * alpha_dot
            omega = omega_ + dt * omega_dot

            self.state = np.array([alpha, omega])

            s = omega

            self.update_feature_list(dt * phi_s_u, s_, s, 0)

            omega_ = omega
            alpha_ = alpha

            # ------------------------------------- Storing the states ----------------------------------------------
            self.alpha_list.append(alpha)
            self.omega_list.append(omega)
            self.state_list.append(np.array([alpha, omega]))
