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

import SEIHRD


# Load a data file
file_name = "control_data/deterministic_mitigation.json"
# file_name = "control_data/deterministic_suppression.json"

num_iters = 10
N_grad_step = 0.0001 / 1000

delta_t = 1
T = 6000

Sigma_0 = np.zeros((6, 1))
Sigma_0[:, 0] = [7.6E6 - 3000, 1000, 2000, 0, 0, 0]

with open(file_name) as json_file:
    results = json.load(json_file)

# Changing beta_init may find different local optima
beta_init = 0.87

for element in results.items():
    if isinstance(element[1], list):
        results[element[0]] = np.asarray(element[1])

# For the first time use:
# results = {}
# results['N'] = 7.6E6
# results['delta_t'] = delta_t
# results['T'] = T
# results['end_time'] = T / delta_t

# results['beta'] = np.zeros(2) # beta will be initialized as beta_init
# results['end_time']=4990

# results['alpha'] = 0.192
# results['b'] = 0.87
# results['gamma_0'] = 0.189
# results['lambda_0'] = 0.0264
# results['delta_0'] = 0.002
# results['delta_1'] = 0.031
# results['gamma_1'] = 0.1
# results['Sigma'] = Sigma_0
# results['k'] = 100
# results['c'] = 3500
# results['d'] = 1000000
# results['mu'] = 0.01

results = SEIHRD.solve_SEIHRD(results, beta_init=beta_init,
                              num_iters=num_iters,
                              N_grad_step=N_grad_step)

name = file_name
print(name)

for element in results.items():
    if isinstance(element[1], np.ndarray):
        results[element[0]] = element[1].tolist()

with open(name, "w") as write_file:
    json.dump(results, write_file, indent=4,
              separators=(", ", ": "), sort_keys=True)

for element in results.items():
    if isinstance(element[1], list):
        results[element[0]] = np.asarray(element[1])

SEIHRD.plot_SEIHRD(results)
# SEIHRD.plot_sim(results)
