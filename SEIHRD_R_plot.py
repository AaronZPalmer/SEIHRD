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

from SEIHRD import SEIHRD

seihrd = SEIHRD()

T = 10000
seihrd.delta_t = 1

time_steps = round(T / seihrd.delta_t)

seihrd.mf[0] = seihrd.N - 5000
seihrd.mf[1] = 2000
seihrd.mf[2] = 3000
Sigma_0 = seihrd.mf
print(Sigma_0)

R = np.arange(0.5, 2, 0.01)

gamma_tilde = seihrd.gamma_0 + seihrd.lambda_0 + seihrd.delta_0

beta = R * gamma_tilde

Sigma_T = np.zeros((6, R.size))

for i in range(0, R.size):

    seihrd.mf = Sigma_0
    seihrd.beta = beta[i]

    for t in range(1, time_steps + 1):

        seihrd.update_mf()
    Sigma_T[:, i] = seihrd.mf
    print(R[i])

fig, ax = plt.subplots(figsize=(7, 5))
print('Plotting')


ax.plot(R, np.minimum(seihrd.N, gamma_tilde * seihrd.N / beta), color='red')
ax.plot(R, Sigma_T[0, :], color='goldenrod', linestyle='dashed')

ax.set_xlabel(r'$R_0$')
ax.set_ylabel('population size')

plt.legend((r'$\max\{N,\frac{ N}{R_0}\}$', r'$S_T$'))

ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.show()
