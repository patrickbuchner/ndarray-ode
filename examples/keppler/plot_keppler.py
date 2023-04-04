#!/usr/bin/python
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import sys
method = sys.argv[1]

folder = 'ndarray_ode/examples/keppler/'
# Import netCDF file
file = folder + "keppler_implicit.parquet"
data = pq.read_table(file)
var = data.to_pandas()
# Prepare Data to Plot
T_im = var['t'][:]
x_im = var['x'][:]
y_im = var['y'][:]
px_im = var['px'][:]
py_im = var['py'][:]
# Import netCDF file
file = folder + "keppler_explicit.parquet"
data = pq.read_table(file)
var = data.to_pandas()
# Prepare Data to Plot
T_ex = var['t'][:]
x_ex = var['x'][:]
y_ex = var['y'][:]
px_ex = var['px'][:]
py_ex = var['py'][:]
# Import netCDF file
file = folder + "keppler_symplectic.parquet"
data = pq.read_table(file)
var = data.to_pandas()
# Prepare Data to Plot
T_sy = var['t'][:]
x_sy = var['x'][:]
y_sy = var['y'][:]
px_sy = var['px'][:]
py_sy = var['py'][:]


# Use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Actual plotting
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(x_im, y_im, label="Implicit")
ax1.plot(x_ex, y_ex, label="Explicit")
ax1.plot(x_sy, y_sy, label="Symplectic")
ax1.set_aspect("equal")
ax1.set(xlabel=r'X', ylabel=r'Y', title=r'Trajectory')
fig1.suptitle(method, fontsize=16)

# Other options
plt.legend(fontsize=12)
plt.grid()
# plt.savefig(folder + "plot.png", dpi=300)
plt.show()
