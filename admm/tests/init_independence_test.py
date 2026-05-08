import torch
from admm.layers import ADMM_Linear
from admm.activations import ADMM_ReLU
from admm.dataclasses import ADMMLayerConfig

def test_weight_bias_initialization_independence():
    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')
    batch, in_f, out_f = 16, 32, 32
    config = ADMMLayerConfig(rho=1.0, beta=1.0, deltas=0.8, thetas=1.0, use_bias=False)

    a_prev = torch.randn((batch, in_f), dtype=torch.float64)
    z_target = torch.randn((batch, out_f), dtype=torch.float64)

    print("\n" + "="*65)
    print("  PROOF A: PURE W UPDATE OVERWRITE (BIAS = FALSE)")
    print("="*65)
    
    layer_z_nobias = ADMM_Linear(in_f, out_f, h=ADMM_ReLU(), config=config)
    layer_r_nobias = ADMM_Linear(in_f, out_f, h=ADMM_ReLU(), config=config)
    layer_z_nobias._setup()
    layer_r_nobias._setup()

    layer_z_nobias.W.data = torch.zeros_like(layer_z_nobias.W)
    layer_r_nobias.W.data = torch.randn_like(layer_r_nobias.W) * 100.0

    layer_z_nobias.z = z_target.clone()
    layer_r_nobias.z = z_target.clone()

    # ONE single update
    layer_z_nobias.update_weights(a_prev)
    layer_r_nobias.update_weights(a_prev)

    w_diff_nobias = torch.max(torch.abs(layer_z_nobias.W - layer_r_nobias.W)).item()
    print(f"[{'Weight (W) Divergence':<26}] Max Difference: {w_diff_nobias:.8e} " + ("✅" if w_diff_nobias < 1e-9 else "❌"))


    print("\n" + "="*65)
    print("  PROOF B: COUPLED W/B CONVERGENCE (BIAS = TRUE)")
    print("="*65)

    config = ADMMLayerConfig(rho=1.0, beta=1.0, deltas=0.8, thetas=1.0, use_bias=True)

    layer_z_bias = ADMM_Linear(in_f, out_f, h=ADMM_ReLU(), config=config)
    layer_r_bias = ADMM_Linear(in_f, out_f, h=ADMM_ReLU(), config=config)
    layer_z_bias._setup()
    layer_r_bias._setup()

    layer_z_bias.W.data = torch.zeros_like(layer_z_bias.W)
    layer_z_bias.b.data = torch.zeros_like(layer_z_bias.b)
    
    layer_r_bias.W.data = torch.randn_like(layer_r_bias.W) * 100.0
    layer_r_bias.b.data = torch.randn_like(layer_r_bias.b) * 100.0

    layer_z_bias.z = z_target.clone()
    layer_r_bias.z = z_target.clone()

    # Because W depends on b, and b depends on W, we run a short 
    # alternating loop (micro-iterations) to allow them to wash out.
    for _ in range(10):
        layer_z_bias.update_weights(a_prev)
        layer_z_bias.update_bias(a_prev)
        
        layer_r_bias.update_weights(a_prev)
        layer_r_bias.update_bias(a_prev)

    w_diff_bias = torch.max(torch.abs(layer_z_bias.W - layer_r_bias.W)).item()
    b_diff_bias = torch.max(torch.abs(layer_z_bias.b - layer_r_bias.b)).item()

    print(f"[{'Weight (W) Divergence':<26}] Max Difference: {w_diff_bias:.8e} " + ("✅" if w_diff_bias < 1e-9 else "❌"))
    print(f"[{'Bias (b) Divergence':<26}] Max Difference: {b_diff_bias:.8e} " + ("✅" if b_diff_bias < 1e-9 else "❌"))
    print("="*65 + "\n")