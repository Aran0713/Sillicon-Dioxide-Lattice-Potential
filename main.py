#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

start_time = time.time()

# N = number of points
N = 400
# Number of potentials
P = 1
# Disctance between potentials
D = 10
# L = x goes from -L to L
L = P*(0.5*D)
# h = step size
h = (2.0*L) / N
# Constants
q = 1.0 / (h**2)
u = -0.5*q

# Arrays
V = np.empty([N+1])
M = np.empty([N+1,N+1])
O = np.empty([N+1])

# Filling up values for x
x = np.linspace(-L,L,N+1)

# Filling up values for array V
for i in range (0,N+1):
    V[i] = 0
for j in range (0,P):
    A = -L + (j*D)
    B = A + D
    for i in range (0,N+1):
        if x[i] >= A and x[i] < B:
            const = (A+B) / 2.0
            V[i] = -0.7*(1 + np.tanh((x[i]-const) + 0.8))*(1 + np.tanh(-(x[i]-const) + 0.8))
    
# Filling up values for matrix M
for i in range (0,N+1):
    for j in range (0,N+1):
        if j == i:
            M[i][j] = V[i] + q
        elif j == i-1 or j == i+1:
            M[i][j] = u
        else:
            M[i][j] = 0
            
# Diagonalizing matrix M
spectrum,v=np.linalg.eigh(M)

# Calculating the number of discrete energies
numen = 0
for i in range (0,N+1):
    if spectrum[i] < 0:
        numen = numen + 1
print("\nNumber of discrete energies = ",numen,"\n")

# Printing discrete energies
for i in range (0,numen):
    print("Energy_"+str(i+1)+" = ",spectrum[i])
print("\n")

# Defining function for calculated wavefunction
def calcpsi(j):
    for i in range(0,N+1):
        O[i] = v[i][j-1]
    return O

# Defining function for normalized calculated wavefunction
def psi(j):
    A = 1 / np.sqrt(trapz(calcpsi(j)**2,x))
    for i in range (0,N+1):
        O = A * calcpsi(j)
    return O

#################### Plots ####################

# Plot of potential
plt.plot(x,V,color='k')
for i in range(0,numen):
    plt.hlines(spectrum[i],-L,L,color="gray",linestyle='--')
    plt.plot(x,(psi(i+1)**2)+spectrum[i],lw='0.8')
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Plot of potential, d = '+str(D))
plt.show()

# Plots of wavefunctions
for i in range (0,numen):
    plt.plot(x,psi(i+1))
    plt.xlabel('x')
    plt.ylabel('\N{GREEK CAPITAL LETTER PSI}(x)')
    plt.title('Plot of \N{GREEK CAPITAL LETTER PSI}(x), n = '+str(i+1)+', d = '+str(D)+' , N = '+str(N))
    plt.show()
    

# Plot of energy
for i in range(0,numen):
    plt.plot(i+1,spectrum[i],'bo')
plt.xlim(0,numen+1)
plt.ylim(-1.6,0)
plt.title("Plot of Eigenvalues")
plt.xlabel("n")
plt.ylabel("Eigenvalues")
plt.show()

print("Runtime = ", time.time() - start_time, "seconds")
