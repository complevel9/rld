#!/usr/bin/env python3


import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Plot graphs. May require manual tweaking.")
parser.add_argument('-s', '--save', type=str)
args = parser.parse_args()
save = args.save

plt.rcParams.update({'font.size': 8})


def plot_ema(vals, f=0.1):
	k = 1 - f
	ema = []
	expsum = 0
	for i in range(len(vals)):
		expsum *= k
		expsum += vals[i]
		ema.append(expsum / (1-k**(i+1)) * (1-k))
	plt.plot(ema, linewidth=0.5)

def plot_sma(vals, n=8):
	sma = []
	nsum = 0
	for i in range(len(vals)):
		nsum += vals[i]
		if i-n >= 0:
			nsum -= vals[i-n]
		sma.append(nsum / min(n,i+1))
	plt.plot(sma, linewidth=0.5)

def plot2_split(vals, labela=None, labelb=None):
	a, b = vals[0::2], vals[1::2]
	plt.plot(a, linewidth=0.5, color='orange', label=labela)
	plt.plot(b, linewidth=0.5, color='blue', label=labelb)
	if (labela is not None) or (labelb is not None):
		ax = plt.gca()
		ax.legend(loc='best')


# ====================================== Return

plt.figure(figsize=(4, 3))
with open('ret.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		vals = [float(x) for x in row]
		plot_ema(vals, 0.05)

plt.xlabel("Episodes")
# plt.ylabel("Mean return")
plt.ylabel("Return")
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(which='major')
plt.tight_layout()

if save:
	name = save + '-ret.pdf'
	plt.savefig(name)
	print('saved ' + name)
else:
	plt.show()

# ====================================== Action values

plt.figure(figsize=(4, 3))
with open('theta.csv') as csvfile:
	reader = csv.reader(csvfile)
	# labels = ['Top', 'Bottom']
	for row in reader:
		vals = [float(x) for x in row]
		# plot2_split(vals, *labels)
		plot2_split(vals)
		# labels = [None, None]

plt.xlabel("Episodes")
plt.ylabel("Action value estimate")
# plt.yticks(np.arange(0, 1.1, 0.2))
plt.grid(which='major')
plt.tight_layout()

if save:
	name = save + '-theta.pdf'
	plt.savefig(name)
	print('saved ' + name)
else:
	plt.show()

# ====================================== G^{-1} diagonals

plt.figure(figsize=(4, 3))
with open('metric.csv') as csvfile:
	reader = csv.reader(csvfile)
	# labels = ['Top', 'Bottom']
	for row in reader:
		vals = [float(x) for x in row]
		# plot2_split(vals, *labels)
		plot2_split(vals)
		# labels = [None, None]

plt.xlabel("Episodes")
plt.ylabel("$\\mathbf{\\hat G}^{-1}$ diagonal components")
plt.gca().set_yscale('log', base=10)
# plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(which='major')
plt.tight_layout()

if save:
	name = save + '-metric.pdf'
	plt.savefig(name)
	print('saved ' + name)
else:
	plt.show()
