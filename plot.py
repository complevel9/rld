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



# ====================================== Mean Return (MC)

# colors = ["black", "red", "green", "blue"]
# labels = ['RG', 'Natural RG', 'Natural RG-FG (error clipping)', 'Natural RG-FAEG']
labels = ['Sarsa(λ)', 'Natural Sarsa(λ)', 'Natural Sarsa(λ)-FG (error clipping)', 'Natural Sarsa(λ)-FAEG']
ntrials = 2000

if True:
	plt.figure(figsize=(4, 3))
	with open('ret.csv') as csvfile:
		reader = csv.reader(csvfile)
		irow = 0
		avg = 0
		for row in reader:
			vals = np.array([float(x) for x in row])
			avg += vals
			if irow % ntrials == ntrials - 1:
				avg = avg / ntrials
				plt.plot(avg, linewidth=0.5, label=labels[irow // ntrials])
				avg = 0
			irow += 1

	plt.xlabel("Episodes")
	plt.ylabel("Mean return")
	# plt.ylabel("Return")
	plt.grid(which='major')
	plt.gca().legend(loc='lower right')
	plt.tight_layout()

	if save:
		name = save + '-ret.pdf'
		plt.savefig(name)
		print('saved ' + name)
	else:
		plt.show()


# ====================================== Return (Flip)

if False:
	plt.figure(figsize=(4, 3))
	with open('ret.csv') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			vals = [float(x) for x in row]
			plot_ema(vals, 0.05)

	plt.xlabel("Episodes")
	# plt.ylabel("Mean return")
	# plt.ylabel("Return")
	plt.ylabel("Undiscounted full return")
	# plt.yticks(np.arange(0, 1.1, 0.1))
	plt.grid(which='major')
	plt.tight_layout()

	if save:
		name = save + '-ret.pdf'
		plt.savefig(name)
		print('saved ' + name)
	else:
		plt.show()

# ====================================== Action values

if False:
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

if False:
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
