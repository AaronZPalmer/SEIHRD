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
import matplotlib.pyplot as plt
import matplotlib

file_name = "control_data/stochastic_suppression.json"

with open(file_name) as json_file:
    results = json.load(json_file)

for element in results.items():
    print(element[0], type(element[1]))
    if isinstance(element[1], list):
        results[element[0]] = np.asarray(element[1])

# file_name_2 = "control_data/vaccine_suppression.json"
file_name_2 = "control_data/discrete_suppression.json"

with open(file_name_2) as json_file:
    results_2 = json.load(json_file)

for element in results_2.items():
    print(element[0], type(element[1]))
    if isinstance(element[1], list):
        results_2[element[0]] = np.asarray(element[1])
#  We now run the simulation

delta_x = results['delta_x']
N = results['N']

state_size = (int)(1 / delta_x) + 1

fig, ax = plt.subplots(figsize=(8, 5))
# i = np.arange(1, 101)
# s = state_size - 101

s = np.arange(0, state_size - 101)
i = 2

I = N * delta_x * i
S = N * delta_x * s

beta = results['beta']
beta_2 = results_2['beta']

# ax.plot(I, beta[s + state_size * i])
# ax.plot(I, beta_2[s + state_size * i + 1 * state_size * state_size])

ax.plot(S, beta[s + state_size * i])
ax.plot(S, beta_2[s + state_size * i + state_size * state_size])
ax.set_ylim((0, 0.9))
ax.set_ylabel(r'$\beta$')
# ax.set_xlabel(r'$\tilde{I}$')
ax.set_xlabel(r'$\tilde{S}$')
# ax.legend(('no vaccine', r'with vaccine, $u=1$'))
ax.legend((r'optimal continuous $\beta$', r'optimal discrete $\beta$'))
ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.show()
