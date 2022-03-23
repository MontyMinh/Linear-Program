import numpy as np
import random
from scipy.optimize import linprog
import pandas as pd
from numpy.testing import assert_almost_equal as aae
from sympy import *
from scipy.linalg import block_diag

def build_dataframe(prog, no_I, no_J):
    
    df = pd.DataFrame(data = np.round(prog.x, 9).reshape(no_I, no_J), columns=[f'Customer {i}' for i in range(no_J)])
    df.index.name = 'Factory'
    
    return df

class CompletionError(Exception):
    
    def __init__(self):
        self.message = 'CompletionError is raised to stop the program'
        super().__init__(self.message)
        
def verify_completion(prog, stop = False):
    
    if prog.status == 0:
        print(f"\033[1;32m{prog.message}\033[0;0m", 
              '\N{party popper}\N{party popper}\N{party popper}')

    elif stop and prog.status !=0:
        print(f'\033[1;31m{prog.message}\033[0;0m', 
              '\N{warning sign}\N{warning sign}\N{warning sign}')
        raise CompletionError
    else:
        print(f'\033[1;31m{prog.message}\033[0;0m')

def generate_objective_vector(no_I, no_J, no_K, no_P, ibik, obij, vci):
    vci = np.tile(vci.reshape(no_P, no_I, 1), reps = no_J)
    c = np.concatenate([ibik.reshape(no_P, no_I, no_K), 
                        (obij+vci).reshape(no_P, no_I, no_J)], axis = 2).flatten()
    return c

def generate_demand_matrix(no_I, no_J, no_K, no_P):
    
    sub_block = np.tile(np.hstack([np.zeros((no_J, no_K)), np.eye(no_J)]), reps = no_I)
    
    Wijk = block_diag(*[sub_block]*no_P)
    
    return Wijk

def generate_capacity_matrix(no_I, no_J, no_K, no_P, ti):

    product_block_list = []
    for p in range(no_P):
        
        factory_block_list = []
        port_block = np.zeros((no_I, no_K))
        for i in range(no_I):
            customer_column = np.zeros((no_I, 1))
            customer_column[i] = ti[p][i]
            customer_block = np.repeat(customer_column, repeats = no_J, axis = 1)
            factory_block_list.append(np.hstack([port_block, customer_block]))
            
        product_block_list += factory_block_list

    Ki = np.concatenate(product_block_list, axis = 1)
    
    return Ki

def generate_supply_matrix_with_efficiency(no_I, no_J, no_K, no_P, ei):

    left = np.tile(-ei.reshape(no_P*no_I, 1), reps = no_K)

    right = np.ones((no_P*no_I, no_J))

    Si = block_diag(*np.hstack([left, right]))
    
    return Si

def optimize_logistics(no_I, no_J, no_K, no_P, ibik, obij, vci, wj, ki, ei, ti):
    
    assert np.sum(ki) >= np.sum(wj), 'More Demand than Capacity. Program is Impossible'

    ### Standard form of our model
    ## Assume non-trivial
    Wijk = generate_demand_matrix(no_I, no_J, no_K, no_P) # Demand Constraint Matrix
    Ki = generate_capacity_matrix(no_I, no_J, no_K, no_P, ti) # Capacity Constraint Matrix

    # Sufficient Supply Matrix
    Si = generate_supply_matrix_with_efficiency(no_I, no_J, no_K, no_P, ei)

    # New cost vector
    c = generate_objective_vector(no_I, no_J, no_K, no_P, ibik, obij, vci)

    # Upper Bound
    A_ub = np.vstack([-Wijk, Ki, Si])
    b_ub = np.hstack([-wj.flatten(), ki.flatten(), np.zeros(no_I*no_P)])

    prog = linprog(c, A_ub = A_ub, b_ub = b_ub) # Bigger than or equal constraints

    result = prog.x.reshape(no_P, no_I, no_J+no_K)
    
    ### Check demand, note that the axis = 1 the array is 3d
    aae(np.sum(result[:, :, no_K:], axis = 1), wj)

    ### Check weighted inbound volume vs outbound volume
    aae(np.sum(result[:, :, :no_K], axis = 2)*ei, np.sum(result[:, :, no_K:], axis = 2))
    
    return prog