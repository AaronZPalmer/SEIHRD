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

T = 365
seihrd.delta_t = 1

time_steps = round(T / seihrd.delta_t)

Sigma = np.zeros((6, time_steps + 1))

seihrd.mf[0] = seihrd.N - 5000
seihrd.mf[1] = 2000
seihrd.mf[2] = 3000
Sigma[:, 0] = seihrd.mf

# seihrd.beta = 4 * (seihrd.lambda_0 + seihrd.gamma_0 + seihrd.delta_0)
# above is for R = 4, below sets R = 0/5
seihrd.beta = (seihrd.lambda_0 + seihrd.gamma_0 + seihrd.delta_0) / 2

print(seihrd.beta)

for t in range(1, time_steps + 1):

    seihrd.update_mf()

    Sigma[:, t] = seihrd.mf

times = seihrd.delta_t * np.arange(0, time_steps + 1)

# fig, ax = plt.subplots(figsize=(9, 5))
# ax.plot(times, Sigma[0].transpose(), color='goldenrod')
# ax.plot(times, Sigma[1], color='mediumorchid')
# ax.plot(times, Sigma[2], color='deepskyblue')
# ax.plot(times, Sigma[3], color='mediumseagreen')
# ax.plot(times, Sigma[4], color='salmon')
# ax.plot(times, Sigma[5], color='slategrey')
# ax.set_xlabel('time (days)')
# ax.set_ylabel('population size')
# plt.legend(('S', 'E', 'I', 'H', 'R', 'D'))
# ax.get_yaxis().set_major_formatter(
#     matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

fig, axs = plt.subplots(2, figsize=(9, 5))

axs[0].plot(times, Sigma[0].transpose(), color='goldenrod')
axs[0].set_ylabel('population size')
axs[1].plot(times, Sigma[1], color='mediumorchid')
axs[1].plot(times, Sigma[2], color='deepskyblue')
axs[1].plot(times, Sigma[3], color='mediumseagreen')
axs[1].plot(times, Sigma[4], color='salmon')
axs[1].plot(times, Sigma[5], color='slategrey')


axs[1].set_xlabel('time (days)')
axs[1].set_ylabel('population size')
axs[0].legend(('S'))
axs[1].legend(('E', 'I', 'H', 'R', 'D'))

axs[0].get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
axs[1].get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.show()
