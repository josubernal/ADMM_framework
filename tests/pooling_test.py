"""
ADMM Pooling Operator Verification (Dot Product Test)

PURPOSE: 
In ADMM, we solve subproblems using 'adjoint operators' to project errors backward. 
This script verifies that the 'adjoint' method of our pooling layers is mathematically correct.

THE MATH:
A forward operator (A) and its adjoint (A^T) must satisfy:
    torch.sum(A(x) * y) == torch.sum(x * A_T(y))

"""

import torch
from admm.pooling import ADMM_Flatten, ADMM_GAP, ADMM_SpatialPool

def test_pooling_adjoint_math():
    test_shape = (4, 16, 32, 32) 
    operators = [ADMM_Flatten(), ADMM_GAP(), ADMM_SpatialPool(output_size=(4, 4))]
    print("\n" + "="*55)
    print("  POOLING OPERATOR MATH TEST")
    print("="*55)
    
    for op in operators:
        x = torch.randn(test_shape, dtype=torch.float64)
        Ax = op(x)
        y = torch.randn_like(Ax, dtype=torch.float64)
        A_T_y = op.adjoint(y, original_input_shape=x.shape)
        
        inner_1 = torch.sum(Ax * y)
        inner_2 = torch.sum(x * A_T_y)
        difference = torch.abs(inner_1 - inner_2).item()
        if difference < 1e-9 :
            print(f"PASSED: The adjoint  {op.__class__.__name__} logic is mathematically perfect.✅")
        assert difference < 1e-9, f"Adjoint failed for {op.__class__.__name__}"
   
    print("\n=======================================================\n")