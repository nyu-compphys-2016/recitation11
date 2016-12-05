
# coding: utf-8

# In[47]:

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pickle

# In[69]:

class Burger:

    Nxtot = 10
    Nx = 10
    Ng = 1
    a = 0.0
    b = 1.0
    dx = 0.1
    t = 0.0
    cfl = 0.5
    x = None
    u = None
    comm = None
    rank = -1
    size = -1
    
    def __init__(self, Nx, a, b, t, cfl):
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.Nxtot = Nx

        self.Nx = int(self.Nxtot / self.size) + 2*self.Ng
        H = (b-a) / float(self.size)

        self.a = a + self.rank*H
        self.b = a + (self.rank+1)*H
        self.t = t
        self.dx = (self.b-self.a) / float(self.Nx - 2*self.Ng)

        self.u = np.zeros(self.Nx)
        self.x = self.a + self.dx*(np.arange(self.Nx)-self.Ng+0.5)
        self.cfl = cfl
    
    def setInitialCondition(self, f,*args):
        self.u = f(self.x, *args)
        
    def plot(self, ax=None, filename=None):
        
        if self.rank == 0:
            if ax is None:
                fig, ax = plt.subplots(1,1)
            else:
                fig = ax.get_figure()
                
            ax.plot(self.x, self.u, 'k+')
            ax.set_xlabel(r'$X$')
            ax.set_ylabel(r'$U$')
            ax.set_title("t = "+str(self.t))
            
            if filename is not None:
                fig.savefig(filename)
            
        return ax
    
    def evolve(self, tfinal):
        
        while self.t < tfinal:
            # Get dt for this timestep.
            # Don't go past tfinal!
            dt = self.getDt()
            if self.t + dt > tfinal:
                dt = tfinal - self.t
                
            #Calculate fluxes
            udot = self.Lu()
            
            #update u
            self.u[:] += dt*udot
            self.synchronize()
            self.t += dt
            print(self.t)
            
    def getDt(self):
        return self.cfl * self.dx / np.fabs(self.u).max()
    
    def Lu(self):
        
        ap = np.empty(self.Nx-1)
        am = np.empty(self.Nx-1)
        
        for i in range(self.Nx-1):
            ap[i] = max(0, self.u[i], self.u[i+1])
            am[i] = max(0,-self.u[i],-self.u[i+1])
            
        F = 0.5*self.u*self.u
        FL = F[:-1]
        FR = F[1:]
        uL = self.u[:-1]
        uR = self.u[1:]
        
        FHLL = (ap*FL + am*FR - ap*am*(uR-uL)) / (ap+am)
        
        LU = np.zeros(self.Nx)
        LU[1:-1] = -(FHLL[1:] - FHLL[:-1]) / self.dx
        
        return LU

    def synchronize(self):

        left = self.rank - 1
        right = self.rank + 1
        
        if left == -1:
            left = self.size-1
        if right == self.size:
            right = 0

        iLL = 0
        iL = self.Ng
        iR = self.Nx - 2*self.Ng
        iRR = self.Nx - self.Ng

        self.comm.Sendrecv(self.u[iR:iR+self.Ng], right, 0,
                        self.u[iLL:iLL+self.Ng], left, 0)
        self.comm.Sendrecv(self.u[iL:iL+self.Ng], left, 0,
                        self.u[iRR:iRR+self.Ng], right, 0)
    
    def saveTxt(self, filename):
        f = open(filename, "w")
        
        f.write(str(self.Nx)+"\n")
        f.write(str(self.a)+"\n")
        f.write(str(self.b)+"\n")
        f.write(str(self.t)+"\n")
        f.write(str(self.cfl)+"\n")
        f.write(" ".join([str(x) for x in self.x]) + "\n")
        f.write(" ".join([str(u) for u in self.u]) + "\n")
    
        f.close()
        
    def loadTxt(self, filename):
        f = open(filename, "r")
        
        self.Nx = int(f.readline())
        self.a = float(f.readline())
        self.b = float(f.readline())
        self.t = float(f.readline())
        self.cfl = float(f.readline())
        self.dx = (self.b-self.a) / (float(self.Nx))
        
        x_str = f.readline()
        u_str = f.readline()
            
        f.close()
        
        self.x = np.array([float(x) for x in x_str.split()])
        self.u = np.array([float(u) for u in u_str.split()])
    
    def savePickle(self, filename):
        f = open(filename, "w")
        pickle.dump(self, f, protocol=-1)
        f.close()
        


# In[26]:

def gauss(x, x0, sigma):
    return np.exp(-(x-x0)*(x-x0) / (2*sigma*sigma))


# In[70]:

if __name__ == "__main__":
    b = Burger(100,-3,4,0,0.5)
    b.setInitialCondition(gauss, 0.0, 1.0)
    b.plot(filename="u.pdf")
    b.evolve(3.0)
    print("Plotting")
    b.plot(filename="u2.pdf")

