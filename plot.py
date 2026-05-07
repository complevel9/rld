#!/usr/bin/env python3


import csv
import matplotlib.pyplot as plt

# sl = [-462, -294, -230, -208, -215, -180, -182, -187, -183, -175, -198, -176, -188, -187, -187][:10]
# nsl = [-306, -227, -226, -198, -184, -172, -194, -183, -166, -164, -162, -168, -152, -160, -154][:10]
# plt.plot(sl, label="Sarsa(λ)")
# plt.plot(nsl, label="Natural Sarsa(λ)")

# rg = [-567, -613, -401, -388, -404, -324, -296, -330, -261, -295, -250, -228, -219, -232, -211][:10]
# nrg = [-570, -523, -379, -442, -379, -314, -291, -285, -291, -290, -280, -230, -270, -249, -240][:10]

# plt.plot(rg, label="RG")
# plt.plot(nrg, label="Natural RG")


# with open('flip.csv', newline='') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for row in reader:
# 		mean_ret = [float(x) for x in row[0:]]
# 		plt.plot(mean_ret, label='idk')

plt.rcParams.update({'font.size': 8})


plt.figure(figsize=(4, 3))
ax = plt.gca()

with open('flip.csv', newline='') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		rets = [float(x) for x in row[0:]]
		k = 0.99
		expavg_rets = []
		expsum = 0
		for i in range(len(rets)):
			expsum *= k
			expsum += rets[i]
			expavg_rets.append(expsum / (1-k**(i+1)) * (1-k))
		plt.plot(expavg_rets, linewidth=0.5)


plt.title("RG ($\\alpha=0.01$) on Flip(2000)")

plt.xlabel("Episodes")
# ax.ylabel("Mean return")
plt.ylabel("EMA return")

# ax.legend(loc='lower right')

plt.grid(which='major')

for tick in ax.get_xticklabels():
    tick.set_fontsize(8)
for tick in ax.get_yticklabels():
    tick.set_fontsize(8)

plt.tight_layout()

plt.show()
# plt.savefig('flip2k-20rg-ema0.01.pdf')
