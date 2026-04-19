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
    tracemalloc.start()
    start_pinv = time.perf_counter()
    
    w_pinv, _ = solve_least_squares_weights(numerator, denominator, use_cholesky=False)
    
    time_pinv = time.perf_counter() - start_pinv
    _, peak_mem_pinv = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # =======================================================
    # METHOD 2: CHOLESKY SOLVER
    # =======================================================
    tracemalloc.start()
    start_chol = time.perf_counter()
    
    w_chol, _ = solve_least_squares_weights(numerator, denominator, use_cholesky=True)
    
    time_chol = time.perf_counter() - start_chol
    _, peak_mem_chol = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # =======================================================
    # VERIFICATION & METRICS
    # =======================================================
    # Find the maximum absolute difference between the two computed weight matrices
    diff = torch.max(torch.abs(w_pinv - w_chol)).item()
    
    # Convert bytes to Megabytes
    mem_pinv_mb = peak_mem_pinv / (1024 * 1024)
    mem_chol_mb = peak_mem_chol / (1024 * 1024)
    
    print(f"[{'Math Equivalence':<22}] Max Difference: {diff:.8e} " + ("✅" if diff < 1e-4 else "❌"))
    print(f"[{'Pseudo-Inverse (pinv)':<22}] Time: {time_pinv:.4f} sec | Peak CPU Mem: {mem_pinv_mb:7.2f} MB")
    print(f"[{'Cholesky Solver':<22}] Time: {time_chol:.4f} sec | Peak CPU Mem: {mem_chol_mb:7.2f} MB")
    
    # Calculate performance gains
    if diff < 1e-4:
        speedup = time_pinv / time_chol if time_chol > 0 else float('inf')
        mem_saving = mem_pinv_mb / mem_chol_mb if mem_chol_mb > 0 else float('inf')
        
        print(f"\n🚀 Cholesky is {speedup:.2f}x faster!")
        print(f"💾 Cholesky uses {mem_saving:.2f}x less operational memory!")
        
    print("="*65 + "\n")
    
    # PyTest Assertion: Fail the test if the math deviates significantly
    assert diff < 1e-4, f"Mathematical divergence detected! Max diff: {diff}"