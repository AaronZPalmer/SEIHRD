import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import SEIHRD


# Load value function results and then plot simulations
# file_name = "control_data/vaccine_suppression.json"
# file_name = "control_data/stochastic_mitigation.json"
file_name = "control_data/stochastic_suppression.json"

with open(file_name) as json_file:
    results = json.load(json_file)

for element in results.items():
    print(element[0], type(element[1]))
    if isinstance(element[1], list):
        results[element[0]] = np.asarray(element[1])

b = results['b']
delta_x = results['delta_x']
N = results['N']
k = results['k']
d = results['d']
c = results['c']
o_1 = results['o_1']
w_0 = results['w_0']

beta = results['beta']
v = results['v']

state_size = (int)(1 / delta_x) + 1

num_sim = 5
delta_t = 0.1

# We define a starting distribution
Sigma_start = np.zeros((6, num_sim))

exposed_start = 2E4
infected_start = 3E4

Sigma_start[0, :] = N - exposed_start - infected_start
Sigma_start[1, :] = exposed_start
Sigma_start[2, :] = infected_start

si = SEIHRD.SEIHRD(N=N, num_sim=num_sim, delta_t=delta_t,
                   b=b, k=k, d=d, c=c, Sigma_0=Sigma_start[:, 0],
                   o_1=o_1, w_0=w_0)

# End time
T = 365
time_steps = round(T / si.delta_t)

control_cost = np.zeros((time_steps + 1, num_sim))
hospital_cost = np.zeros((time_steps + 1, num_sim))
death_cost = np.zeros((time_steps + 1, num_sim))

cost_to_go = np.zeros((time_steps + 1, num_sim))

Sigma = np.zeros((6, time_steps + 1, num_sim))

beta_sim = np.zeros((time_steps, num_sim))


Sigma[:, 0, :] = Sigma_start

si.J = np.zeros((1, num_sim))
si.m = Sigma_start

#  We now run the simulation

for t in range(1, time_steps + 1):

    # (s,i,u) will be converted to the integer units for evaluating beta / v
    s = np.ceil(si.m[0] / N / delta_x)
    s = s.astype(int)

    i = np.ceil((si.m[1] + si.m[2]) / N / delta_x)
    i = i.astype(int)

    u = si.u.astype(int)

    print(i, u)

    cost_to_go[t - 1] = -v[s + i * state_size + u * state_size * state_size] \
        * np.where(Sigma[2, t - 1] + Sigma[1, t - 1] == 0, 0, 1)

    si.beta = beta[s + state_size * i + u * state_size * state_size]
    beta_sim[t - 1] = si.beta

    si.update_sim()

    control_cost[t] = control_cost[t - 1] + delta_t * (
        - k * N * (np.log(si.beta / si.b) - (
            si.beta - si.b) / si.b))
    hospital_cost[t] = hospital_cost[t - 1] + delta_t * c * \
        (si.m[3] + 0.5 * si.m[3] * si.m[3] / N)
    death_cost[t] = d * si.m[5]

    Sigma[:, t] = si.m

times = np.arange(0, T + delta_t, delta_t)

# fig, axs = plt.subplots(3, figsize=(9, 5))

# axs[0].plot(times, Sigma[0, :, 0], color='goldenrod', label='S')
# axs[0].plot(times, Sigma[0, :, 1:], color='goldenrod', label='_nolegend_')
# axs[0].plot(times, Sigma[1, :, 0], color='mediumorchid', label='E')
# axs[0].plot(times, Sigma[1, :, 1:], color='mediumorchid', label='_nolegend_')
# axs[0].plot(times, Sigma[2, :, 0], color='deepskyblue', label='I')
# axs[0].plot(times, Sigma[2, :, 1:], color='deepskyblue', label='_nolegend_')
# axs[0].plot(times, Sigma[3, :, 0], color='mediumseagreen', label='H')
# axs[0].plot(times, Sigma[3, :, 1:], color='mediumseagreen', label='_nolegend_')
# axs[0].plot(times, Sigma[4, :, 0], color='salmon', label='R')
# axs[0].plot(times, Sigma[4, :, 1:], color='salmon', label='_nolegend_')
# axs[0].plot(times, Sigma[5, :, 0], color='slategrey', label='D')
# axs[0].plot(times, Sigma[5, :, 1:], color='slategrey', label='_nolegend_')
# axs[0].set_ylabel('population size')
# axs[0].legend()
# axs[0].get_yaxis().set_major_formatter(
#     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# times = np.arange(0, T, delta_t)
# axs[1].plot(times, beta_sim, color='purple')

# axs[1].set_xlabel('time')
# axs[1].set_ylim((0, 0.9))
# axs[1].set_ylabel(r'$\beta$')

# times = np.arange(0, T + delta_t, delta_t)

# axs[2].plot(times, control_cost[:, 0] / results['N'],
#             color='blue', label='control cost per person')
# axs[2].plot(times, control_cost[:, 1:] / results['N'],
#             color='blue', label='_nolegend_')

# axs[2].plot(times, hospital_cost[:, 0] / results['N'],
#             color='orange', label='hospital cost per person')
# axs[2].plot(times, hospital_cost[:, 1:] / results['N'],
#             color='orange', label='_nolegend_')
# axs[2].plot(times, death_cost[:, 0] / results['N'],
#             color='green', label='death cost per person')
# axs[2].plot(times, death_cost[:, 1:] / results['N'],
#             color='green', label='_nolegend_')


# axs[2].set_xlabel('time (days)')
# axs[2].set_ylabel('US dollars')
# axs[2].get_yaxis().set_major_formatter(
#     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# axs[2].legend()

# Plot S separately
fig, axs = plt.subplots(4, figsize=(9, 7))

axs[0].plot(times, Sigma[0], color='goldenrod')
axs[0].legend(('S'))
axs[0].set_ylabel('population size')
axs[0].get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
axs[1].plot(times, Sigma[1, :, 0], color='mediumorchid', label='E')
axs[1].plot(times, Sigma[1, :, 1:], color='mediumorchid', label='_nolegend_')
axs[1].plot(times, Sigma[2, :, 0], color='deepskyblue', label='I')
axs[1].plot(times, Sigma[2, :, 1:], color='deepskyblue', label='_nolegend_')
axs[1].plot(times, Sigma[3, :, 0], color='mediumseagreen', label='H')
axs[1].plot(times, Sigma[3, :, 1:], color='mediumseagreen', label='_nolegend_')
axs[1].plot(times, Sigma[4, :, 0], color='salmon', label='R')
axs[1].plot(times, Sigma[4, :, 1:], color='salmon', label='_nolegend_')
axs[1].plot(times, Sigma[5, :, 0], color='slategrey', label='D')
axs[1].plot(times, Sigma[5, :, 1:], color='slategrey', label='_nolegend_')
axs[1].set_ylabel('population size')
axs[1].legend()
axs[1].get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

times = np.arange(0, T, delta_t)
axs[2].plot(times, beta_sim, color='purple')

# axs[2].set_xlabel('time')
axs[2].set_ylim((0, 0.9))
axs[2].set_ylabel(r'$\beta$')

times = np.arange(0, T + delta_t, delta_t)

axs[3].plot(times, control_cost[:, 0] / results['N'],
            color='blue', label='control cost per person')
axs[3].plot(times, control_cost[:, 1:] / results['N'],
            color='blue', label='_nolegend_')

axs[3].plot(times, hospital_cost[:, 0] / results['N'],
            color='orange', label='hospital cost per person')
axs[3].plot(times, hospital_cost[:, 1:] / results['N'],
            color='orange', label='_nolegend_')
axs[3].plot(times, death_cost[:, 0] / results['N'],
            color='green', label='death cost per person')
axs[3].plot(times, death_cost[:, 1:] / results['N'],
            color='green', label='_nolegend_')


axs[3].set_xlabel('time (days)')
axs[3].set_ylabel('US dollars')
axs[3].get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

axs[3].legend()

plt.show()
