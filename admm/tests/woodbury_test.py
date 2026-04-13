"""
ADMM Woodbury Matrix Identity Verification

PURPOSE:
Verifies that using the Woodbury Matrix Identity to bypass the inversion of 
massive spatial matrices produces mathematically identical results to the 
standard dense solver, while drastically reducing memory and compute time.
"""

import torch
import time
from admm.solvers import solve_linear_system, solve_woodbury_system

def test_woodbury_identity():
    print("\n" + "="*60)
    print("  WOODBURY MATRIX IDENTITY VERIFICATION TEST")
    print("="*60)
    
    # Force double precision for strict mathematical verification
    torch.set_default_dtype(torch.float64)
    
    # Simulating the exact bottleneck from your network:
    # A Linear layer receiving a flattened 16-channel 16x16 image
    in_features = 4096  
    out_features = 10   # 10 output classes
    batch_size = 50
    out_shape = (batch_size, in_features)
    
    beta_eff = 0.1
    rho = 0.01
    
    print(f"Simulating Bottleneck: {in_features} Input Features -> {out_features} Output Classes\n")
    
    # Random weights and targets
    W = torch.randn((out_features, in_features), dtype=torch.float64)
    B = torch.randn(out_shape, dtype=torch.float64)
    
    # =======================================================
    # METHOD 1: STANDARD DENSE SOLVER
    # =======================================================
    start_dense = time.perf_counter()
    
    # Explicitly build the massive W^T W matrix (4096 x 4096)
    WtW = torch.matmul(W.t(), W)
    I_dense = torch.eye(in_features, dtype=torch.float64)
    A_dense = beta_eff * I_dense + rho * WtW
    
    # Solve standard system
    x_dense = solve_linear_system(A_dense, B, out_shape, in_features)
    time_dense = time.perf_counter() - start_dense
    
    # =======================================================
    # METHOD 2: WOODBURY SOLVER
    # =======================================================
    start_wood = time.perf_counter()
    
    # Our Woodbury solver handles the inversion internally using the 10x10 trick
    x_wood = solve_woodbury_system(W, B, beta_eff, rho, out_shape, in_features)
    time_wood = time.perf_counter() - start_wood
    
    # =======================================================
    # VERIFICATION & METRICS
    # =======================================================
    diff = torch.max(torch.abs(x_dense - x_wood)).item()
    
    # Calculate memory footprint of the core matrices inverted
    # 64-bit float = 8 bytes per element
    mem_dense_mb = (in_features * in_features * 8) / (1024 * 1024)
    mem_wood_bytes = (out_features * out_features * 8) 
    
    print(f"[{'Math Equivalence':<25}] Max Difference: {diff:.8e} " + ("✅" if diff < 1e-9 else "❌"))
    print(f"[{'Dense Inversion Size':<25}] {in_features}x{in_features} Matrix ({mem_dense_mb:.2f} MB)")
    print(f"[{'Woodbury Inversion Size':<25}] {out_features}x{out_features} Matrix ({mem_wood_bytes} Bytes)")
    
    if diff < 1e-9:
        speedup = time_dense / time_wood if time_wood > 0 else float('inf')
        mem_saving = (mem_dense_mb * 1024 * 1024) / mem_wood_bytes
        
        print(f"\n🚀 Woodbury is {speedup:.2f}x faster!")
        print(f"💾 Woodbury uses {mem_saving:,.0f}x less inversion memory!")
        
    print("="*60 + "\n")
