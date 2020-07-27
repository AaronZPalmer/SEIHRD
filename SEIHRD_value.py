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

import json
import numpy as np


file_name = "control_data/discrete_suppression.json"

N = 7.6E6

beta = 0.1  # This will be controlled
b = 0.87
tilde_gamma = 0.217  # Recovery rate (called tilde_gamma elsewhere)
d = 1000000  # Cost per death
# d = 100000  # for mitigation
c = 3500  # cost per hospitalized

p_D = 0.038  # Percentage of recovered people that die
# p_H = 0.12  # Percentage of infected people that are hospitalized
p_H = 0.202

k = 100  # constant in cost

# For vaccines
# o_1 = 1 / 100  # rate of vaccination
# w_0 = 1 / 100  # rate of discovery of vaccination
o_1 = 0
w_0 = 0


# Either do discrete or continuous
R_values = np.asarray([0.5, 1, 2, 4])  # range of available values for beta
beta_values = R_values * tilde_gamma
b = beta_values[3]

delta_x = 0.001  # spatial discretization
state_size = (int)(1 / delta_x) + 1

state_size_N = N * delta_x - 1

k_N = N * delta_x
gamma_e = 0.5772156649

size = 2 * state_size * state_size
# size = state_size * state_size

# Optimal beta values and the expected value are stored as vectors
beta = b * np.ones(size)
v = np.zeros(size)

error_threshold = 1E-6

for u in range(1, -1, -1):
    print(u)
    for s in range(0, state_size):
        print(s)
        S = s * delta_x  # S is ratio of susceptible population

        v[s + state_size * state_size * u] = - d * p_D * (1 - S)
        for i in range(1, state_size - 1):

            I = i * delta_x  # I is ratio of infected population

            H = p_H * I  # number of hospitalized
            # Coeff will store the coeffitions to v(s,i,i) and
            # rhs will store other values
            # this will be vectors of size beta_values

            # We adjust tilde_gamma to better handle small I values
            tilde_gamma_old = tilde_gamma
            if i == 1:
                tilde_gamma = tilde_gamma / (
                    np.log(k_N) + gamma_e + 1 / 2 / k_N - 1 / 12 / k_N / k_N)
            else:
                diff_sum = np.log(I / (I - delta_x)) \
                    + delta_x / 2 / k_N * (1 / I - 1 / (I - delta_x)) \
                    - delta_x * delta_x / 12 / k_N / k_N * (
                        1 / I / I - 1 / (I - delta_x) / (I - delta_x))
                tilde_gamma = delta_x / I * tilde_gamma / diff_sum

            # The cost is subtracted from the RHS
            rhs = -delta_x * k * (
                beta_values / b - np.log(beta_values / b) - 1)
            rhs = rhs - delta_x * c * (H + 0.5 * H * H)
            # Coeff and cost have terms to reflect the recovery transition
            coeff = tilde_gamma * I
            rhs = rhs + tilde_gamma * I * v[
                s + state_size * (i - 1) + state_size * state_size * u]

            # If s>0 terms are added for infection and vaccination transition
            if s > 0:
                rhs = rhs + beta_values * S * I * v[
                    s - 1 + state_size * (i + 1) + state_size * state_size * u]
                coeff = coeff + beta_values * S * I

            # If u==0 a term is added for the vaccination discover transition
            if u == 0:
                rhs = rhs + w_0 * delta_x * v[
                    s + state_size * i + 1 * state_size * state_size]
                coeff = coeff + w_0 * delta_x
            else:
                if s > 0:
                    rhs = rhs + o_1 * v[
                        (s - 1) + state_size * i + 1 * state_size * state_size]
                    coeff = coeff + o_1

            # temporary values of v are stored for each choice of beta
            temp_v = rhs / coeff

            # beta is chosen to maximize these values
            v_arg = np.argmax(temp_v)
            v[s + state_size * i + u * state_size * state_size] = temp_v[v_arg]
            beta[s + state_size * i + state_size * state_size * u] = \
                beta_values[v_arg]

            tilde_gamma = tilde_gamma_old


results = {}
results['beta'] = beta
results['v'] = v
results['delta_x'] = delta_x

results['N'] = N

results['b'] = b
results['tilde_gamma'] = tilde_gamma   # Recovery rate
results['d'] = d  # Cost per death
results['c'] = c  # cost per hospitalized

results['p_D'] = p_D  # Percentage of recovered people that die
# p_H = 0.12  # Percentage of infected people that are hospitalized
results['p_H'] = p_H

results['k'] = k  # constant in cost

# For vaccines
results['o_1'] = o_1  # rate of vaccination
results['w_0'] = w_0  # rate of discovery of vaccination


name = file_name
print(name)

for element in results.items():
    if isinstance(element[1], np.ndarray):
        results[element[0]] = element[1].tolist()

with open(name, "w") as write_file:
    json.dump(results, write_file, indent=4,
              separators=(", ", ": "), sort_keys=True)
