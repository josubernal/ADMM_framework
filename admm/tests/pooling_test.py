import torch
from ..pooling import ADMM_Flatten, ADMM_GAP, ADMM_SpatialPool

def verify_adjoint_math(operator, input_shape):
    """
    Performs the dot product test: <Ax, y> == <x, A^T y>
    """
    print(f"--- Testing {operator.__class__.__name__} ---")
    
    x = torch.randn(input_shape, dtype=torch.float64)
    Ax = operator(x)
    y = torch.randn_like(Ax, dtype=torch.float64)
    A_T_y = operator.adjoint(y, original_input_shape=x.shape)
    inner_product_1 = torch.sum(Ax * y)
    inner_product_2 = torch.sum(x * A_T_y)
    difference = torch.abs(inner_product_1 - inner_product_2).item()
    
    print(f"<Ax, y>    = {inner_product_1.item():.8f}")
    print(f"<x, A^T y> = {inner_product_2.item():.8f}")
    print(f"Difference  = {difference:.8e}")
    
    if difference < 1e-9:
        print("✅ PASSED: The adjoint logic is mathematically perfect.\n")
    else:
        print("❌ FAILED: There is a mismatch. Check the upsampling/scaling logic.\n")

# ==========================================
# Run the Tests
# ==========================================
if __name__ == "__main__":
    test_shape = (4, 16, 32, 32) 
    
    verify_adjoint_math(ADMM_Flatten(), test_shape)
    verify_adjoint_math(ADMM_GAP(), test_shape)
    verify_adjoint_math(ADMM_SpatialPool(output_size=(4, 4)), test_shape)