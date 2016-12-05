from mpi4py import MPI
import numpy as np

def f(x):
    return x*np.sin(x)*np.sin(x)

def riemann_sum(x, f):
    N = x.shape[0]
    h = (x[-1] - x[0]) / (N-1)
    return h * f(x[:-1]).sum()

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


a = 0.0
b = 10.0

Ntot = 1024

N = int(Ntot / size)
H = (b-a) / float(size)

xa = a + rank*H
xb = a + (rank+1)*H

x = np.linspace(xa, xb, N)

result = riemann_sum(x, f)
res_arr = np.array(result, dtype=np.float64)

total = np.array([0.0], dtype=np.float64)
total2 = np.zeros(size, dtype=np.float64)

comm.Reduce(res_arr, total, MPI.SUM, 0)
comm.Scatter([total,MPI.DOUBLE], [total2,MPI.DOUBLE], 0)

#print("Hi! I am process {0:d} of {1:d}! My result is {2:g} and total is {3:g}"
#        .format(rank, size, res_arr[0], total[0]))

print((rank, res_arr, total,total2))

#if rank == 0:
#    print("Total is: " + str(total[0]))




