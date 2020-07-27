# Copyright 2020 Aaron Zeff Palmer

# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:

# 1. Redistributions of source code must retain the above
# copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# The SEIHRD class simulates the SEIHRD model
# It can either run the simulation as a differential equation
# or the simulation as a stochastic process.
# For the differential equation it allows computing gradients

class SEIHRD:

    def __init__(self, N=7.6E6, delta_t=0.1, num_sim=1,
                 beta=0.87, alpha=0.192, b=0.87,
                 gamma_0=0.189, lambda_0=0.0264, delta_0=0.002,
                 delta_1=0.031, gamma_1=0.1,
                 Sigma_0=[7.6E6 - 10, 0, 10, 0, 0, 0], k=100,
                 c=3500, d=1000000, mu=0.01,
                 o_1=0, w_0=0):

        # Paramters:
        self.N = N  # Total population size

        self.delta_t = delta_t  # Time step
        self.num_sim = num_sim  # Number of simulations
        # (for running as a stochastic process)
        self.beta = beta  # Infection rate (this is the control parameter)
        self.alpha = alpha  # Exposed to infected rate
        self.gamma_0 = gamma_0  # Enfected to recovered rate
        self.lambda_0 = lambda_0  # Enfected to hospitalized rate
        self.gamma_1 = gamma_1  # Hospitalized to recovered rate
        self.delta_0 = delta_0  # Infected to dead rate
        self.delta_1 = delta_1  # Hospitalized to dead rate

        self.k = k  # Coefficient for the control cost function
        self.d = d  # Coefficient for the death cost
        self.c = c  # Coefficient for hospitalization cost

        self.mu = mu  # Numerical coefficient for terminal state constraint
        self.b = b  # Base infection rate
        self.w_0 = w_0  # Vaccination discovery rate
        self.o_1 = o_1  # Vaccination dispense rate

        # Initialize the empirical measure for stochastic processes
        self.m = np.asarray(Sigma_0).reshape((6, 1)) \
            * np.ones((1, self.num_sim))

        # Initialize the mean field approximation for the differential equation
        self.mf = np.asarray(Sigma_0)

        # Initialize the costate
        # (for computing gradients of the differential equation)
        self.P = np.zeros(6)

        # Initialize the cost
        self.J = np.zeros(num_sim)
        # Initialize the vaccine variable
        self.u = np.zeros(num_sim)

    # update_mf runs a single forward Euler update
    # for the differential equation
    def update_mf(self):

        self.J = self.J + self.delta_t * self.cost()

        self.mf = self.mf + self.delta_t * self.f()

    # update_sim runs a single step to simulate
    # as a stochastic process
    def update_sim(self):

        # We simulate each transition by a Poisson random variable
        a = np.zeros((9, self.num_sim))

        a[0] = np.random.poisson(
            self.delta_t * self.beta * self.m[0] * self.m[2] / self.N,
            self.num_sim)  # I to E
        a[1] = np.random.poisson(
            self.delta_t * self.alpha * self.m[1], self.num_sim)  # E to I
        a[2] = np.random.poisson(
            self.delta_t * self.lambda_0 * self.m[2], self.num_sim)  # I to H
        a[3] = np.random.poisson(
            self.delta_t * self.gamma_0 * (self.m[2]), self.num_sim)  # I to R
        a[4] = np.random.poisson(
            self.delta_t * self.delta_0 * (self.m[2]), self.num_sim)  # I to D
        a[5] = np.random.poisson(
            self.delta_t * self.gamma_1 * self.m[3], self.num_sim)  # H to R
        a[6] = np.random.poisson(
            self.delta_t * self.delta_1 * (self.m[3]), self.num_sim)  # H to D

        a[7] = np.random.poisson(
            self.delta_t * self.w_0 * (1 - self.u), self.num_sim)
        a[8] = np.random.poisson(
            self.delta_t * self. N * self.o_1 * self.u, self.num_sim)

        # We make sure we never transition more than population avialable
        a[0] = np.minimum(a[0], self.m[0])
        a[1] = np.minimum(a[1], self.m[1])
        a[2] = np.minimum(a[2], self.m[2])
        a[3] = np.minimum(a[3], self.m[2] - a[2])
        a[4] = np.minimum(a[4], self.m[2] - a[2] - a[3])
        a[5] = np.minimum(a[5], self.m[3])
        a[6] = np.minimum(a[6], self.m[3] - a[5])
        a[7] = np.minimum(a[7], 1)
        a[8] = np.minimum(a[8], self.m[0] - a[0])

        rhs = np.zeros((6, self.num_sim))

        rhs[0] = -a[0] - a[8]
        rhs[1] = a[0] - a[1]
        rhs[2] = a[1] - a[2] - a[3] - a[4]
        rhs[3] = a[2] - a[5] - a[6]
        rhs[4] = a[3] + a[5]
        rhs[5] = a[4] + a[6]

        self.J = self.J + self.delta_t * self.cost_sim()

        self.m = self.m + rhs

        self.u = self.u + a[7]

    # update_P computes one backward step of the discretized costate equation
    def update_P(self):

        self.P = self.P + self.delta_t * self.grad_H_state()

    # f is the rhs of the differential equation
    def f(self):
        f = np.zeros(6)

        f[0] = -self.beta * self.mf[0] * self.mf[2] / self.N
        f[1] = self.beta * self.mf[0] * self.mf[2] / self.N \
            - self.alpha * self.mf[1]
        f[2] = self.alpha * self.mf[1] - \
            (self.gamma_0 + self.lambda_0 + self.delta_0) * self.mf[2]
        f[3] = self.lambda_0 * self.mf[2] - \
            (self.gamma_1 + self.delta_1) * self.mf[3]
        f[4] = self.gamma_0 * self.mf[2] + self.gamma_1 * self.mf[3]
        f[5] = self.delta_0 * self.mf[2] + self.delta_1 * self.mf[3]
        return f

    # compute the gradient with respect to the control variable
    def grad_f_control(self):
        grad_f = np.zeros(6)
        grad_f[0] = - self.mf[0] \
            * self.mf[2] / self.N
        grad_f[1] = self.mf[0] \
            * self.mf[2] / self.N

        return grad_f

    # compute the Jacobian gradient with respect to the state variables
    def grad_f_state(self):
        grad_f = np.zeros((6, 6))
        grad_f[0, 0] = -self.beta * self.mf[2] / self.N
        grad_f[0, 2] = -self.beta * self.mf[0] / self.N

        grad_f[1, 0] = self.beta * self.mf[2] / self.N
        grad_f[1, 1] = - self.alpha
        grad_f[1, 2] = self.beta * self.mf[0] / self.N

        grad_f[2, 1] = self.alpha
        grad_f[2, 2] = - (self.gamma_0 + self.lambda_0 + self.delta_0)

        grad_f[3, 2] = self.lambda_0
        grad_f[3, 3] = - (self.gamma_1 + self.delta_1)

        grad_f[4, 2] = self.gamma_0
        grad_f[4, 3] = self.gamma_1

        grad_f[5, 2] = self.delta_0
        grad_f[5, 3] = self.delta_1

        return grad_f

    # compute the cost for the differential equation
    def cost(self):

        cost = - self.k * self.N * \
            (np.log(self.beta / self.b) - (
                self.beta - self.b) / self.b) \
            + self.c * \
            (self.mf[3] + 0.5 * self.mf[3] * self.mf[3] / self.N)
        return cost

    # compute the cost for the stochastic process
    def cost_sim(self):

        cost = - self.k * self.N * \
            (np.log(self.beta / self.b) - (
                self.beta - self.b) / self.b) \
            + self.c * \
            (self.m[3] + 0.5 * self.m[3] * self.m[3] / self.N)
        return cost

    # gradient of the cost with respect to the control
    def grad_cost_control(self):
        grad_cost = - self.k * self.N * (1 / self.beta - 1 / self.b)
        return grad_cost

    # gradient of the cost with respect to the state
    def grad_cost_state(self):
        grad_cost = np.zeros(6)
        grad_cost[3] = self.c * (1 + self.mf[3] / self.N)
        return grad_cost

    # compute the Hamiltonian
    def H(self):

        H = np.inner(self.f(), self.P) - self.cost()

        return H

    # compute the gradient of the Hamiltonian with respect to the control
    def grad_H_control(self):

        grad_H = np.inner(self.grad_f_control(), self.P) \
            - self.grad_cost_control()

        return grad_H

    # compute the gradient of the Hamiltonian with respect to the state
    def grad_H_state(self):
        grad_H = self.grad_f_state().transpose().dot(self.P) \
            - self.grad_cost_state()

        return grad_H

    # compute the total, terminal cost
    def total_cost(self):
        thresh = self.mf[1] + self.mf[2] \
            + self.mf[3]
        cost = self.d * self.mf[5] + self.J + self.N / 2 / self.mu \
            * np.square(np.maximum(thresh - np.exp(-1), 0))
        return cost


# solve_SEIHRD solves for an optimal trajectory
# for the differential SEIRHD model
def solve_SEIHRD(results, beta_init=1,
                 num_iters=100, N_grad_step=0.001,
                 gamma_m=0.9, error_threshold=1E-6):

    # load model with parameters specified in results
    seihrd = SEIHRD(N=results['N'], delta_t=results['delta_t'],
                    alpha=results['alpha'], b=results['b'],
                    gamma_0=results['gamma_0'], lambda_0=results['lambda_0'],
                    delta_0=results['delta_0'],
                    delta_1=results['delta_1'], gamma_1=results['gamma_1'],
                    k=results['k'], c=results['c'], d=results['d'],
                    mu=results['mu'])

    # T is end time and this is the total time steps
    time_steps = round(results['T'] / seihrd.delta_t)

    # num_iters is total number of gradient descent iterations
    grad_step = N_grad_step / seihrd.N  # gradient step

    gamma_m = 0.9  # moment factor

    mf = np.zeros((6, time_steps + 1))  # State (S,E,I,H,R,D)

    end_time = results['end_time']

    P = np.zeros((6, time_steps + 1))  # costate

    # initialize
    v = np.zeros(time_steps)
    beta = results['beta']
    # If sizes do not match, initialize as beta_init
    if beta.size != time_steps:
        beta = beta_init * np.ones(time_steps)
        end_time = time_steps

    H = np.zeros(time_steps)
    grad_H = np.zeros(time_steps)

    total_cost = np.zeros(num_iters)
    control_cost = np.zeros(time_steps + 1)
    hospital_cost = np.zeros(time_steps + 1)
    death_cost = np.zeros(time_steps + 1)
    delta_t = seihrd.delta_t

    lam = np.zeros(num_iters)  # multiplier for target constraint

    Sigma_0 = results['Sigma'][:, 0]

    for i in range(0, num_iters):

        mf[:, 0] = Sigma_0

        seihrd.mf = mf[:, 0]
        seihrd.J = 0

        # Run the forward simulation
        for t in range(1, end_time + 1):

            seihrd.beta = beta[t - 1]

            seihrd.update_mf()

            mf[:, t] = seihrd.mf
            control_cost[t] = control_cost[t - 1] + delta_t * (
                -seihrd.k * seihrd.N * (np.log(seihrd.beta / seihrd.b) - (
                    seihrd.beta - seihrd.b) / seihrd.b))
            hospital_cost[t] = hospital_cost[t - 1] + delta_t * seihrd.c * \
                (seihrd.mf[3] + 0.5 * seihrd.mf[3] * seihrd.mf[3] / seihrd.N)
            death_cost[t] = seihrd.d * seihrd.mf[5]

        thresh = mf[1, end_time] + mf[2, end_time] \
            + mf[3, end_time]

        print(i, ' thresh ', thresh)

        total_cost[i] = seihrd.total_cost()

        # Determine the Lagrange multiplier and costate terminal condtions
        lam[i] = np.maximum(thresh - np.exp(-1), 0) \
            / seihrd.mu * seihrd.N

        print(i, ' lambda', lam[i] / seihrd.N)
        print(i, ' total cost', total_cost[i] / seihrd.N)
        print(i, ' end time', end_time)

        P[0, end_time] = 0
        P[4, end_time] = 0
        P[5, end_time] = -seihrd.d
        P[2, end_time] = -lam[i]
        P[1, end_time] = -lam[i]
        P[3, end_time] = -lam[i]

        seihrd.P = P[:, end_time]
        seihrd.mf = mf[:, end_time - 1]
        seihrd.beta = beta[end_time - 1]

        # Update the end time
        if seihrd.H() > 0 and end_time < time_steps and np.mod(i, 10) == 0:
            end_time = end_time + 1
        else:
            if end_time > 1:

                thresh = mf[1, end_time - 1] + mf[2, end_time - 1] \
                    + mf[3, end_time - 1]
                temp_lam = np.maximum(thresh - np.exp(-1), 0) \
                    / seihrd.mu * seihrd.N
                temp_P = np.zeros(6)
                temp_P[0] = 0
                temp_P[4] = 0
                temp_P[5] = -seihrd.d
                temp_P[2] = -temp_lam
                temp_P[1] = -temp_lam
                temp_P[3] = -temp_lam

                seihrd.P = temp_P
                seihrd.mf = mf[:, end_time - 2]
                seihrd.beta = beta[end_time - 2]
                if seihrd.H() < 0:
                    lam[i] = temp_lam
                    end_time = end_time - 1
                    P[:, end_time] = temp_P

        # Run the backward costate equations,
        # computing the gradient of the control
        for t in range(end_time, 0, -1):

            seihrd.P = P[:, t]
            seihrd.mf = mf[:, t - 1]
            seihrd.beta = beta[t - 1]

            H[t - 1] = seihrd.H()

            grad_H_cur = seihrd.grad_H_control()
            # cut off the gradient if it is too large
            grad_H_cur = np.minimum(grad_H_cur, seihrd.N)
            grad_H_cur = np.maximum(grad_H_cur, -seihrd.N)

            grad_H[t - 1] = beta[t - 1] * grad_H_cur / seihrd.N

            # Use a momentum algorithm for alpha = e^beta
            alpha = np.log(beta[t - 1])
            v[t - 1] = gamma_m * v[t - 1] + \
                grad_step * beta[t - 1] * grad_H_cur
            alpha = alpha + v[t - 1]
            beta[t - 1] = np.exp(alpha)

            seihrd.update_P()

            P[:, t - 1] = seihrd.P

        # If residual is below error threshold terminate
        if (i > 0 and np.amax(
            np.abs(grad_H[:end_time])) < error_threshold and np.abs(
                total_cost[i] - total_cost[i - 1]) < error_threshold):
            num_iters = i + 1
            break

    results['Sigma'] = mf
    results['P'] = P
    results['beta'] = beta
    results['total_cost'] = total_cost
    results['control_cost'] = control_cost
    results['hospital_cost'] = hospital_cost
    results['death_cost'] = death_cost
    results['end_time'] = end_time
    results['H'] = H
    results['grad_H'] = grad_H
    results['lam'] = lam
    results['num_iters'] = num_iters
    results['grad_step'] = grad_step

    results['N'] = seihrd.N
    results['delta_t'] = seihrd.delta_t  # Time step
    results['alpha'] = seihrd.alpha  # exposed to infected rate
    results['gamma_0'] = seihrd.gamma_0  # infected to recovered rate
    results['lambda_0'] = seihrd.lambda_0  # infected to hospitalized rate
    results['gamma_1'] = seihrd.gamma_1  # hospitalized to recovered rate
    results['delta_0'] = seihrd.delta_0  # infected to dead rate
    results['delta_1'] = seihrd.delta_1  # hodpitalized to dead rate
    results['k'] = seihrd.k
    results['d'] = seihrd.d
    results['c'] = seihrd.c
    results['mu'] = seihrd.mu
    results['b'] = seihrd.b

    return results

# Plot the results of SEIHRD solution
def plot_SEIHRD(results):

    end_time = np.minimum(results['end_time'], 365)
    Sigma = results['Sigma']
    beta = results['beta']

    times = results['delta_t'] * np.arange(0, end_time + 1)

    # Uncomment the following to plot with three plots
    # (S with the rest of the state variables)
    fig, axs = plt.subplots(3, figsize=(9, 5))

    axs[0].plot(times, Sigma[0][:end_time + 1], color='goldenrod')
    axs[0].plot(times, Sigma[1][:end_time + 1], color='mediumorchid')
    axs[0].plot(times, Sigma[2][:end_time + 1], color='deepskyblue')
    axs[0].plot(times, Sigma[3][:end_time + 1], color='mediumseagreen')
    axs[0].plot(times, Sigma[4][:end_time + 1], color='salmon')
    axs[0].plot(times, Sigma[5][:end_time + 1], color='slategrey')
    axs[0].set_xlabel('time (days)')
    axs[0].set_ylabel('population size')
    axs[0].legend(('S', 'E', 'I', 'H', 'R', 'D'))
    axs[0].get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    times = results['delta_t'] * np.arange(0, end_time)
    axs[1].plot(times, beta[:end_time], color='purple')
    axs[1].set_ylim((0, 0.9))
    axs[1].set_ylabel(r'$\beta$')

    times = results['delta_t'] * np.arange(0, end_time + 1)
    axs[2].plot(times, results['control_cost'][:end_time + 1]/results['N'])
    axs[2].plot(times, results['hospital_cost'][:end_time + 1]/results['N'])
    axs[2].plot(times, results['death_cost'][:end_time + 1]/results['N'])

    axs[2].set_xlabel('time (days)')
    axs[2].set_ylabel('US dollars')
    axs[2].get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    axs[2].legend(('control cost per person',
      'hospital cost per person', 'death cost per person'))

    # Comment the following to plot S seperate from EIHRD
    # fig, axs = plt.subplots(4, figsize=(9, 7))

    # axs[0].plot(times, Sigma[0][:end_time + 1], color='goldenrod')
    # axs[0].legend(('S'))
    # axs[0].set_ylabel('population size')

    # axs[0].get_yaxis().set_major_formatter(
    #     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    # axs[1].plot(times, Sigma[1][:end_time + 1], color='mediumorchid')
    # axs[1].plot(times, Sigma[2][:end_time + 1], color='deepskyblue')
    # axs[1].plot(times, Sigma[3][:end_time + 1], color='mediumseagreen')
    # axs[1].plot(times, Sigma[4][:end_time + 1], color='salmon')
    # axs[1].plot(times, Sigma[5][:end_time + 1], color='slategrey')
    # # axs[1].set_xlabel('time (days)')
    # axs[1].set_ylabel('population size')
    # axs[1].legend(('E', 'I', 'H', 'R', 'D'))
    # axs[1].get_yaxis().set_major_formatter(
    #     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # times = results['delta_t'] * np.arange(0, end_time)
    # axs[2].plot(times, beta[:end_time], color='purple')
    # axs[2].set_ylim((0, 0.9))
    # # axs[2].set_xlabel('time')
    # axs[2].set_ylabel(r'$\beta$')

    # times = results['delta_t'] * np.arange(0, end_time + 1)
    # axs[3].plot(times, results['control_cost'][:end_time + 1] / results['N'])
    # axs[3].plot(times, results['hospital_cost'][:end_time + 1] / results['N'])
    # axs[3].plot(times, results['death_cost'][:end_time + 1] / results['N'])

    # axs[3].set_xlabel('time (days)')
    # axs[3].set_ylabel('US dollars')
    # axs[3].get_yaxis().set_major_formatter(
    #     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # axs[3].legend(('control cost per person',
    #                'hospital cost per person', 'death cost per person'))

    plt.show()

# Plot the results of stochastic simulations
def plot_sim(results):

    si = SEIHRD(N=results['N'], delta_t=results['delta_t'],
                alpha=results['alpha'], b=results['b'],
                gamma_0=results['gamma_0'], lambda_0=results['lambda_0'],
                delta_0=results['delta_0'],
                delta_1=results['delta_1'], gamma_1=results['gamma_1'],
                k=results['k'], c=results['c'], d=results['d'],
                mu=results['mu'])

    # T is end time and this is the total time steps
    time_steps = round(results['T'] / si.delta_t)

    beta = results['beta']
    N = results['N']
    d = results['d']

    num_sim = si.num_sim  # number of simulations

    # We define a starting distribution

    delta_t = si.delta_t  # time step

    # End time
    T = 365
    time_steps = round(T / si.delta_t)

    J = np.zeros((time_steps + 1, num_sim))

    Sigma = np.zeros((6, time_steps + 1, num_sim))

    si.J = np.zeros((1, num_sim))

    for t in range(1, time_steps):

        si.beta = beta[t]

        si.update_sim()

        Sigma[:, t] = si.m

        J[t] = si.J / N
        J[t] = J[t] + np.where(
            Sigma[2, t] + Sigma[1, t] == 0, d * Sigma[5, t] / N, 0)

    fig, axs = plt.subplots(5)

    times = np.arange(0, T + delta_t, delta_t)

    # Plot the susceptible population
    axs[0].plot(times, Sigma[0, :])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('susceptible population size')

    # Plot the infected population
    axs[1].plot(times, Sigma[2, :])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('infected population size')

    # Plot beta values
    axs[2].plot(times[0:times.size - 1], beta[0:times.size - 1])
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('beta')

    # Plot the accrued cost
    axs[3].plot(times, J)
    axs[3].set_ylabel('accrued cost')
    axs[3].set_xlabel('time')

    # Plot hospitalized
    axs[4].plot(times, Sigma[3, :])
    axs[4].set_ylabel('hospitalized')
    axs[4].set_xlabel('time')

    plt.show()
