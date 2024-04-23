# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:01:54 2024

@author: vahed
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt
from sympy import symbols, diff, Matrix

# Parameters
E = 30e6 #Young's modulus (psi)
G = 12e6 #Shearing modulus for the beam material (psi)
L = 14 #Overhang length of the member (inch)
tau_max = 13600 #Design stress of the weld (psi)
sigma_max = 30000 #Design normal stress for the beam material (psi)
delta_max = 0.25 #Maximum deflection (inch)
P = 6000 #Load (lb)
C1 = 0.10471 #Cost per unit volume of the weld material ($ inch-3)
C2 = 0.04811 #Cost per unit volume of the bar ($ inch-3)
C3 = 1 #Labor cost per unit weld volume ($ inch-3)
Nfeval = 1
Objective_history=[]
n_variables= 4 # Number of variables


# Objective Function
def objective(x):
    x1, x2, x3, x4 = x
    V_weld = x1**2 * x2 # Volume of weld material (inch3)
    V_bar = x3 * x4 * (L + x2) # Volume of bar (inch3)
    f = (C1+C3) * V_weld + C2 * V_bar #Total material cost to be minimized.
    return f

# Constraints
def constraint1(x):
    # The shear stress at the beam support location cannot exceed
    # maximum allowable for the material
    x1, x2, x3, x4 = x
    # Primary stress acting over the weld throat
    tau_d = P /(np.sqrt(2)*x1*x2)
    
    # Moment of P about center of gravity of weld setup 
    M = P * (L + x2/2)
    R = np.sqrt((x2**2)/4 + ((x1+x3)/2)**2)
    
    # Polar moment of inertia of weld
    J = 2 * (x1*x2 * np.sqrt(2)* ((x2**2)/12 + ((x1+x3)/2)**2))
    
    # Secondary torsional stress.
    tau_dd= M*R/J 
    
    # Weld stress
    tau = np.sqrt(tau_d**2 + 2* tau_d * tau_dd * x2/2/R + tau_dd**2)
    return -(tau - tau_max)

def constraint2(x):
    # The normal bending stress at the beam support location cannot exceed
    #the maximum yield strength for the material
    x1, x2, x3, x4 = x
    # Bar bending stress
    sigma = 6 * P * L /(x4 * x3**2)
    return -(sigma - sigma_max)

def constraint3(x):
    # Side Constraint
    # The member thickness is greater than the weld thickness
    x1, x2, x3, x4 = x

    return -(x1 - x4)

def constraint4(x):
    # Side Constraint
    # The member thickness is greater than the weld thickness
    x1, x2, x3, x4 = x

    return -(C1 * x1**2 + C2 * x3*x4 * (L+ x2) - 5)

def constraint5(x):
    # Side Constraint
    # The weld thickness must be larger than a defined minimum
    x1, x2, x3, x4 = x

    return -(0.125 - x1)

def constraint6(x):
    # The deflection cannot exceed the maximum deflection
    x1, x2, x3, x4 = x
    
    # Bar Deflection. To calculate the deflection, assume the bar to be a cantilever of length L
    delta = 4 * P * L**3 /(E * x4 * x3**3)
    return -(delta - delta_max)

def constraint7(x):
    # The buckling load is greater than the applied load
    x1, x2, x3, x4 = x
    
    # For narrow rectangular bars, the bar buckling load is approximated by
    Pc = 4.013 * E / L**2 * np.sqrt(x3**2 * x4**6 /36)* (1 - x3/2/L * np.sqrt(E/4/G))
    return -(P - Pc)

### Problem's Gradients
# Define symbols
x1, x2, x3, x4 = symbols('x1 x2 x3 x4')

# objective_derivative
V_weld = x1**2 * x2 # Volume of weld material (inch3)
V_bar = x3 * x4 * (L + x2) # Volume of bar (inch3)
f = (C1+C3) * V_weld + C2 * V_bar #Total material cost to be minimized.
df_dx1 = diff(f, x1)
df_dx2 = diff(f, x2)
df_dx3 = diff(f, x3)
df_dx4 = diff(f, x4)
df= [df_dx1, df_dx2, df_dx3, df_dx4]

def objective_derivative(x):
    xx1, xx2, xx3, xx4 = x
    fprime= np.zeros(n_variables)  
    # fprime= np.array([2*(C1+C3) * xx1 * xx2,
    #                   (C1+C3) * xx1**2 + C2 * xx3  * xx4,
    #                   C2 * xx4 * (L + xx2),
    #                   C2 * xx3 * (L + xx2)])
    for i in range(n_variables):
        fprime[i] = df[i].subs({x1:xx1, x2: xx2, x3:xx3, x4:xx4})

    return fprime

# Constraint1_derivative
tau_d = P /(np.sqrt(2)*x1*x2)
# Moment of P about center of gravity of weld setup 
M = P * (L + x2/2)
R = ((x2**2)/4 + ((x1+x3)/2)**2)**0.5
# Polar moment of inertia of weld
J = 2 * (x1*x2 * np.sqrt(2)* ((x2**2)/12 + ((x1+x3)/2)**2))
# Secondary torsional stress.
tau_dd= M*R/J 
# Weld stress
tau = (tau_d**2 + 2* tau_d * tau_dd * x2/2/R + tau_dd**2)**0.5
g1= tau - tau_max
dg1_dx1 = diff(g1, x1)
dg1_dx2 = diff(g1, x2)
dg1_dx3 = diff(g1, x3)
dg1_dx4 = diff(g1, x4)
dg1= [dg1_dx1, dg1_dx2, dg1_dx3, dg1_dx4]

def constraint1_derivative(x):
    xx1, xx2, xx3, xx4 = x
    gprime= np.zeros(n_variables)  
    for i in range(n_variables):
        gprime[i] = dg1[i].subs({x1:xx1, x2: xx2, x3:xx3, x4:xx4})

    return -gprime

# Constraint2_derivative
sigma = 6 * P * L /(x4 * x3**2)
g2= sigma - sigma_max
dg2_dx1 = diff(g2, x1)
dg2_dx2 = diff(g2, x2)
dg2_dx3 = diff(g2, x3)
dg2_dx4 = diff(g2, x4)
dg2= [dg2_dx1, dg2_dx2, dg2_dx3, dg2_dx4]

def constraint2_derivative(x):
    xx1, xx2, xx3, xx4 = x
    gprime= np.zeros(n_variables)  
    for i in range(n_variables):
        gprime[i] = dg2[i].subs({x1:xx1, x2: xx2, x3:xx3, x4:xx4})

    return -gprime

# Constraint3_derivative
g3= x1 - x4
dg3_dx1 = diff(g3, x1)
dg3_dx2 = diff(g3, x2)
dg3_dx3 = diff(g3, x3)
dg3_dx4 = diff(g3, x4)
dg3= [dg3_dx1, dg3_dx2, dg3_dx3, dg3_dx4]

def constraint3_derivative(x):
    xx1, xx2, xx3, xx4 = x
    gprime= np.zeros(n_variables)  
    for i in range(n_variables):
        gprime[i] = dg3[i].subs({x1:xx1, x2: xx2, x3:xx3, x4:xx4})

    return -gprime

# Constraint4_derivative
g4= C1 * x1**2 + C2 * x3*x4 * (L+ x2) - 5
dg4_dx1 = diff(g4, x1)
dg4_dx2 = diff(g4, x2)
dg4_dx3 = diff(g4, x3)
dg4_dx4 = diff(g4, x4)
dg4= [dg4_dx1, dg4_dx2, dg4_dx3, dg4_dx4]

def constraint4_derivative(x):
    xx1, xx2, xx3, xx4 = x
    gprime= np.zeros(n_variables)  
    for i in range(n_variables):
        gprime[i] = dg4[i].subs({x1:xx1, x2: xx2, x3:xx3, x4:xx4})

    return -gprime

# Constraint5_derivative
g5= 0.125 - x1
dg5_dx1 = diff(g5, x1)
dg5_dx2 = diff(g5, x2)
dg5_dx3 = diff(g5, x3)
dg5_dx4 = diff(g5, x4)
dg5= [dg5_dx1, dg5_dx2, dg5_dx3, dg5_dx4]

def constraint5_derivative(x):
    xx1, xx2, xx3, xx4 = x
    gprime= np.zeros(n_variables)  
    for i in range(n_variables):
        gprime[i] = dg5[i].subs({x1:xx1, x2: xx2, x3:xx3, x4:xx4})

    return -gprime

# Constraint6_derivative
delta = 4 * P * L**3 /(E * x4 * x3**3)
g6= delta - delta_max
dg6_dx1 = diff(g6, x1)
dg6_dx2 = diff(g6, x2)
dg6_dx3 = diff(g6, x3)
dg6_dx4 = diff(g6, x4)
dg6= [dg6_dx1, dg6_dx2, dg6_dx3, dg6_dx4]

def constraint6_derivative(x):
    xx1, xx2, xx3, xx4 = x
    gprime= np.zeros(n_variables)  
    for i in range(n_variables):
        gprime[i] = dg6[i].subs({x1:xx1, x2: xx2, x3:xx3, x4:xx4})

    return -gprime

# Constraint7_derivative
Pc = 4.013 * E / L**2 * (x3**2 * x4**6 /36)**0.5 * (1 - x3/2/L * (E/4/G)**0.5)
g7= P - Pc
dg7_dx1 = diff(g7, x1)
dg7_dx2 = diff(g7, x2)
dg7_dx3 = diff(g7, x3)
dg7_dx4 = diff(g7, x4)
dg7= [dg7_dx1, dg7_dx2, dg7_dx3, dg7_dx4]

def constraint7_derivative(x):
    xx1, xx2, xx3, xx4 = x
    gprime= np.zeros(n_variables)  
    for i in range(n_variables):
        gprime[i] = dg7[i].subs({x1:xx1, x2: xx2, x3:xx3, x4:xx4})

    return -gprime

# Checking the status of each constraint according to its sign by standard form.
def cons_check(x,cons):
    constraint1,constraint2,constraint3, constraint4, constraint5, constraint6, constraint7 =cons
    print('Constraint Evaluation:')
    g1 = constraint1(x)
    print('g1 =',-g1)
    if g1<0:
        print('Constraint1 is violated')
    elif g1>0:
        print('Constraint1 is inactive')
    else:
        print('Constraint1 is active')
        
    g2 = constraint2(x)
    print('g2 =',-g2)
    if g2<0:
        print('Constraint2 is violated')
    elif g2>0:
        print('Constraint2 is inactive')
    else:
        print('Constraint2 is active')
        
    g3 = constraint3(x)
    print('g3 =',-g3)
    if g3<0:
        print('Constraint3 is violated')
    elif g3>0:
        print('Constraint3 is inactive')
    else:
        print('Constraint3 is active')
        
    g4 = constraint4(x)
    print('g4 =',-g4)
    if g4<0:
        print('Constraint4 is violated')
    elif g4>0:
        print('Constraint4 is inactive')
    else:
        print('Constraint4 is active')

    g5 = constraint5(x)
    print('g5 =',-g5)
    if g5<0:
        print('Constraint5 is violated')
    elif g5>0:
        print('Constraint5 is inactive')
    else:
        print('Constraint5 is active')
        
    g6 = constraint6(x)
    print('g6 =',-g6)
    if g6<0:
        print('Constraint6 is violated')
    elif g6>0:
        print('Constraint6 is inactive')
    else:
        print('Constraint6 is active')

    g7 = constraint7(x)
    print('g7 =',-g7)
    if g7<0:
        print('Constraint7 is violated')
    elif g7>0:
        print('Constraint7 is inactive')
    else:
        print('Constraint7 is active')
    
    print('===================================')

       
# Callback function to display the optimizer's progress
def callbackf(x):
    global Nfeval
    global Objective_history
    print ('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}  {4: 3.6f}  {5: 3.6f}'.format(Nfeval, x[0], x[1], x[2], x[3], objective(x)))
    Objective_history.append([Nfeval, objective(x)])
    Nfeval += 1
    # print(f"Current solution: {x} Objective: {objective(x)}")


# Bounds
bounds = [(0.1, 2), (0.1, 10), (0.1, 10), (0.1,2)]

# Constraints dictionary
cons = [{'type': 'ineq', 'fun': constraint1, 'jac': constraint1_derivative},
        {'type': 'ineq', 'fun': constraint2, 'jac': constraint2_derivative},
        {'type': 'ineq', 'fun': constraint3, 'jac': constraint3_derivative},
        {'type': 'ineq', 'fun': constraint4, 'jac': constraint4_derivative},
        {'type': 'ineq', 'fun': constraint5, 'jac': constraint5_derivative},
        {'type': 'ineq', 'fun': constraint6, 'jac': constraint6_derivative},
        {'type': 'ineq', 'fun': constraint7, 'jac': constraint7_derivative}]
cons2 = [constraint1, constraint2, constraint3,constraint4, constraint5,constraint6
         ,constraint7]
# Initial guess
x0 = [0.18, 4, 9, 0.22]

# Optimize
print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}  {5:9s}'.format('Iter', ' X1', ' X2', ' X3', 'X4', 'f(X)'))  

result = minimize(objective, x0, method='SLSQP',jac= objective_derivative, bounds=bounds, constraints=cons, callback=callbackf, tol = 1e-6)

print('x*=',result.x)
print('f(x*)=',result.fun)
print('Number of Iterations:',result.nit)
print('Number of Function Evaluations:',result.nfev)

x_o= result.x
cons_check(x_o,cons2)


# Part b
x01 = [[0.3, 5, 8, 0.3],
       [0.8, 1, 7, 1.5],
       [0.2, 3, 9, 0.5]]

fig,ax=plt.subplots(1,1)
for i in range(3):
    Nfeval = 1
    Objective_history=[]
    print('Results of Set', i+1)
    print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}  {5:9s}'.format('Iter', ' X1', ' X2', ' X3', 'X4', 'f(X)'))  
    result = minimize(objective, x01[i], method='SLSQP',jac= objective_derivative, bounds=bounds, constraints=cons, callback=callbackf)
    print('x*=',result.x)
    print('f(x*)=',result.fun)
    print('Number of Iterations:',result.nit)
    print('Number of Function Evaluations:',result.nfev)
    Iter = [item[0] for item in Objective_history]
    f_values =  [item[1] for item in Objective_history]
    plt.plot(Iter , f_values, label=f"run {i+1}")
    
    x_o= result.x
    cons_check(x_o,cons2)
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
plt.legend()
# Save the plot as SVG (high quality, vector format)
plt.savefig('Convergencegraph.png', dpi=300)

# Part c Differential Evolution
# No lower limit on constr_fun
lb =  0

# Upper limit on constr_fun
ub= np.inf

# Nonlinear Constraints
nlc1 = NonlinearConstraint(constraint1, lb, ub, jac= constraint1_derivative)
nlc2 = NonlinearConstraint(constraint2, lb, ub, jac= constraint2_derivative)
nlc3 = NonlinearConstraint(constraint3, lb, ub, jac= constraint3_derivative)
nlc4 = NonlinearConstraint(constraint4, lb, ub, jac= constraint4_derivative)
nlc5 = NonlinearConstraint(constraint5, lb, ub, jac= constraint5_derivative)
nlc6 = NonlinearConstraint(constraint6, lb, ub, jac= constraint6_derivative)
nlc7 = NonlinearConstraint(constraint7, lb, ub, jac= constraint7_derivative)
Cons_diff = [nlc1, nlc2, nlc3, nlc4, nlc5, nlc6, nlc7]
print('Results of differential_evolution')
sol = differential_evolution(objective, bounds=bounds, constraints=Cons_diff, tol= 1e-10, maxiter=8000)
print('x*=',sol.x)
print('f(x*)=',sol.fun)
print('Number of Function Evaluations:',sol.nfev)
x_o= sol.x
cons_check(x_o,cons2)

# Part d
seeds= [157, 4921, 753]

for i in range(3):
    print('Results of differential_evolution set', i+1)
    sol = differential_evolution(objective, bounds=bounds, seed = seeds[i], constraints=[nlc1, nlc2, nlc3, nlc4, nlc5, nlc6, nlc7], tol= 1e-10, maxiter=8000)
    print('x*=',sol.x)
    print('f(x*)=',sol.fun)
    print('Number of Generations:',sol.nit)
    print('Number of Function Evaluations:',sol.nfev)
    x_o= sol.x
    cons_check(x_o,cons2)

