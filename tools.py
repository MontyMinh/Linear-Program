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