#!/usr/bin/python
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import sys
method = sys.argv[1]

folder = 'examples/keppler/'
# Import netCDF file
file = folder + "keppler.parquet"
data = pq.read_table(file)
var = data.to_pandas()

# Use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Prepare Data to Plot
T = var['t'][:]
x = var['x'][:]
y = var['y'][:]
px = var['px'][:]
py = var['py'][:]

# Actual plotting
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(x,y)
ax1.set_aspect("equal")
ax1.set(xlabel=r'X', ylabel=r'Y', title=r'Trajectory')
fig1.suptitle(method, fontsize=16)

# Other options
plt.legend(fontsize=12)
plt.grid()
# plt.savefig(folder + "plot.png", dpi=300)
plt.show()
