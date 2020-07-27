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

file_name_1 = "control_data/deterministic_suppression.json"
file_name_2 = "control_data/deterministic_mitigation.json"


with open(file_name_1) as json_file:
    results_1 = json.load(json_file)

with open(file_name_2) as json_file:
    results_2 = json.load(json_file)

for element in results_1.items():
    if isinstance(element[1], list):
        results_1[element[0]] = np.asarray(element[1])

for element in results_2.items():
    if isinstance(element[1], list):
        results_2[element[0]] = np.asarray(element[1])


end_time_1 = np.minimum(results_1['end_time'], 365)
end_time_2 = np.minimum(results_2['end_time'], end_time_1 * 2)
S_1 = results_1['Sigma'][0, :end_time_1]
S_2 = results_2['Sigma'][0, :end_time_2]

beta_1 = results_1['beta'][:end_time_1]
beta_2 = results_2['beta'][:end_time_2]

N = results_1['N']
lambda_0 = results_1['lambda_0']
gamma_0 = results_1['gamma_0']
delta_0 = results_1['delta_0']

R_1 = beta_1 * S_1 / N / (lambda_0 + gamma_0 + delta_0)
R_2 = beta_2 * S_2 / N / (lambda_0 + gamma_0 + delta_0)

times_1 = results_1['delta_t'] * np.arange(0, end_time_1)
times_2 = results_2['delta_t'] * np.arange(0, end_time_2)

fig, ax = plt.subplots()

ax.plot(times_1, R_1)
ax.plot(times_2, R_2)
ax.plot(times_2, np.ones(end_time_2), color='grey', linestyle='--')

ax.set_xlabel('time (days)')
ax.set_ylabel(r'$R_e$')
ax.legend(('suppression strategy', 'mitigation strategy'))

plt.show()
