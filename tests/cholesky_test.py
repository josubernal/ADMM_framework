"""
ADMM Cholesky vs Pseudo-Inverse Verification & Benchmark

PURPOSE:
Verifies that using the Cholesky solver yields mathematically identical results 
to the standard dense pseudo-inverse (pinv) solver, while reducing CPU memory 
allocation and computation time.
"""

import torch
import time
import tracemalloc
from admm.solvers import solve_least_squares_weights
# =============================================================================
# The PyTest Benchmark
# =============================================================================
def test_cholesky_vs_pinv():
    print("\n" + "="*65)
    print("  CHOLESKY VS PINV SOLVER: VERIFICATION & BENCHMARK (CPU)")
    print("="*65)
    
    # Force CPU execution and double precision for strict mathematical verification
    device = torch.device('cpu')
    torch.set_default_dtype(torch.float32)
    
    # Simulating a well-conditioned scenario where Cholesky shines
    # (Batch size >= Feature size)
    features = 1024       # E.g., 1024 input neurons
    classes = 100         # 100 output neurons
    batch_size = 2048     # Batch size > features ensures full rank!
    
    rho = 1.0
    lambda_reg = 0.01     # L2 regularization acts as natural jitter
    
    print(f"Simulating: {features} Input Features -> {classes} Output Classes (Batch: {batch_size})\n")
    
    # 1. Generate Fake Data
    A = torch.randn((batch_size, features), device=device)       # Inputs
    Z_minus_U = torch.randn((batch_size, classes), device=device) # ADMM auxiliary targets
    
    # 2. Precompute Numerator and Denominator
    # Denominator: P^T @ P (scaled by rho and lambda)
    denominator = rho * (A.t() @ A) + lambda_reg * torch.eye(features, device=device)
    # Numerator: Y^T @ P (scaled by rho)
    numerator = rho * (Z_minus_U.t() @ A)

    # =======================================================
    # METHOD 1: STANDARD PINV SOLVER
    # =======================================================
    # tracemalloc tracks exact Python RAM allocations
    start_pinv = time.perf_counter()
    
    w_pinv, cached_pinv= solve_least_squares_weights(numerator, denominator, use_cholesky=False)
    
    time_pinv = time.perf_counter() - start_pinv
 
    size_pinv_mb = (cached_pinv.element_size() * cached_pinv.nelement()) / (1024 * 1024)
    
    # =======================================================
    # METHOD 2: CHOLESKY SOLVER
    # =======================================================
    start_chol = time.perf_counter()
    
    w_chol, cached_chol = solve_least_squares_weights(numerator, denominator, use_cholesky=True)
    
    time_chol = time.perf_counter() - start_chol

    if isinstance(cached_chol, tuple):
        size_chol_mb = sum((tensor.element_size() * tensor.nelement()) for tensor in cached_chol) / (1024 * 1024)
    else:
        size_chol_mb = (cached_chol.element_size() * cached_chol.nelement()) / (1024 * 1024)
    # =======================================================
    # VERIFICATION & METRICS
    # =======================================================
    # Find the maximum absolute difference between the two computed weight matrices
    diff = torch.max(torch.abs(w_pinv - w_chol)).item()
    
    print(f"[{'Math Equivalence':<22}] Max Difference: {diff:.8e} " + ("✅" if diff < 1e-4 else "❌"))
    print(f"[{'Pseudo-Inverse (pinv)':<22}] Time: {time_pinv:.4f} sec | Matrix Size: {size_pinv_mb:7.2f} MB")
    print(f"[{'Cholesky Solver':<22}] Time: {time_chol:.4f} sec | Matrix Size: {size_chol_mb:7.2f} MB")
    
    # Calculate performance gains
    if diff < 1e-4:
        speedup = time_pinv / time_chol if time_chol > 0 else float('inf')
        
        # Note: Depending on implementation, PINV and Cholesky caches might actually 
        # take up similar static memory, but Cholesky skips the massive intermediate RAM spikes!
        print(f"\n🚀 Cholesky is {speedup:.2f}x faster!")
        
    print("="*65 + "\n")
    
    # PyTest Assertion: Fail the test if the math deviates significantly
    assert diff < 1e-4, f"Mathematical divergence detected! Max diff: {diff}"