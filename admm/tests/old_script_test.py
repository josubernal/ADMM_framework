import torch
import configparser
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import random
import torch.nn as nn
import time
import os

from admm import  ADMM_SpikingLinear, ADMM, ADMM_Heaviside, ADMM_Metrics
from admm.tests.old_script.admm_snn import ADMM_SNN

def test_old_match():
    seed = 8281003564
    def reset_seed(s):
        random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = configparser.ConfigParser()
    base_path = os.path.dirname(__file__)
    config_path = os.path.join(base_path, 'config/config_test.ini')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file at: {config_path}")
    config.read(config_path)
    use_double = config.getboolean('affine', 'use_double', fallback=False)
    
    if use_double:
        torch.set_default_dtype(torch.float64)
        current_dtype = torch.float64
        print("Running in DOUBLE (64-bit) precision.")
    else:
        torch.set_default_dtype(torch.float32)
        current_dtype = torch.float32
        print("Running in SINGLE (32-bit) precision.")  
   
    epochs = config.getint('affine', 'epochs')
    warming_iters = config.getint('affine', 'warming_iters')
    batch_size = config.getint('affine', 'batch_size')
    rho = config.getfloat('affine', 'rho')
    beta = config.getfloat('affine', 'beta')
    
    
    hidden_dims = config.getint('affine', 'hidden_dims')
    thetas = config.getfloat('affine', 'thetas')   
    layers = nn.ModuleList([
        ADMM_SpikingLinear(in_f=34*34*2, out_f=hidden_dims, h=ADMM_Heaviside(thetas=thetas)),
        ADMM_SpikingLinear(in_f=hidden_dims, out_f=10)
    ])

    n_timesteps = config.getint('affine', 'n_timesteps')
    deltas = config.getfloat('affine', 'deltas')
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=1000)
    ])

    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
    train_loader = DataLoader(cached_trainset, batch_size=batch_size,
                              collate_fn=tonic.collation.PadTensors(), shuffle=True, drop_last=True, 
                              generator=torch.Generator().manual_seed(seed))
    
    model = ADMM(layers, T=n_timesteps, rho=rho, thetas=thetas, deltas=deltas, beta=beta, 
                 init="zeros-rng", bias=False, train_method="unrolled-random").to(device)
    
    metrics=ADMM_Metrics(model)
    reset_seed(seed)
    data, targets = next(iter(train_loader)) 
    data, targets = data.to(device), targets.to(device)
    data = data.view(data.size(0), data.size(1), -1)
    
    targets_oop = torch.nn.functional.one_hot(targets.to(torch.long), num_classes=10).to(current_dtype)
    targets_leg = targets_oop.t()
    
    if data.size(1) > n_timesteps:
        data = data[:, :n_timesteps, :]
    base_data = data.permute(1, 0, 2).to(current_dtype)

    # ==========================================
    # OOP MODEL EXECUTION
    # ==========================================
    reset_seed(seed) 
    
    model._init_states(base_data) 
    data_oop = base_data + 0.01 * torch.randn_like(base_data) 
    oop = {}

    start_time_oop = time.time()
    for epoch in range(epochs + 1):
        if epoch > 0:
            model.fit(data_oop, targets_oop, warming=epoch <= warming_iters) 
            
        if epoch % 5 == 0:
            with torch.no_grad():
                raw_outputs, firing_rates = model.forward_model(data_oop)
                flat_outputs = raw_outputs.view(batch_size, -1)
                _, preds = flat_outputs.max(dim=1)
                acc = 100. * (preds == targets_oop.argmax(dim=1)).sum().item() / batch_size
                current_metrics = metrics.get_all_metrics(data_oop, targets_oop)

            oop[epoch] = {
            'W0': model.layers[0].W.detach().clone(),
            'W1': model.layers[1].W.detach().clone(),
            'z0': model.layers[0].z.detach().clone(),
            'a0': model.layers[0].a.detach().clone(),
            'z1': model.layers[1].z.detach().clone(),
            'lam': model.lambda_lagrange.detach().clone(),
            'acc': acc, 
            'mse': current_metrics['mse'], 
            'lagr': current_metrics['lagrangian_cost'], 
            'primal': current_metrics['primal_residual'],
            'pre': current_metrics['preactivation_constraint_sum'], 
            'act': current_metrics['activation_constraint_sum'], 
            'firing_rates': firing_rates, 
            'lam_sum': model.lambda_lagrange.norm().item() / (model.lambda_lagrange.size(0) * model.lambda_lagrange.size(1))
        }

    end_time_oop = time.time()
    time_oop = end_time_oop - start_time_oop
    # ==========================================
    # LEGACY MODEL EXECUTION
    # ==========================================
    reset_seed(seed) 
    
    # Note: Legacy script maps rho->rho (preact penalty) and rho->beta (act penalty)
    model_real = ADMM_SNN(batch_size, n_timesteps, 34*34*2, [hidden_dims], 10, rho, deltas, thetas, beta)
    data_leg = base_data + 0.01 * torch.randn_like(base_data) 
    leg = {}
    
    start_time_leg = time.time()
    for epoch in range(epochs + 1):
        if epoch > 0:
            model_real.fit(data_leg, targets_leg, warming=epoch <= warming_iters) 
            
        if epoch % 5 == 0:
            with torch.no_grad():
                pot, firing_rates_leg = model_real.forward_model(data_leg)
                _, preds_leg = (pot.view(batch_size, -1)).max(1)
                acc_leg = 100. * (preds_leg == targets_oop.argmax(dim=1)).sum().item() / batch_size
                
                mse_leg = model_real.loss(targets_leg).item()
                lagr_leg = model_real.lagrangian_cost(data_leg, targets_leg)
                if isinstance(lagr_leg, torch.Tensor): lagr_leg = lagr_leg.item()
                primal_leg = model_real.primal_residual_norm()
                if isinstance(primal_leg, torch.Tensor): primal_leg = primal_leg.item()
                
                pre_leg = [v.item() if isinstance(v, torch.Tensor) else v for v in model_real.preactivation_constraint_sum(data_leg)]
                act_leg = [v.item() if isinstance(v, torch.Tensor) else v for v in model_real.activation_constraint_sum()]
                lam_sum_leg = torch.norm(model_real.lambda_lagrange).item() / (model_real.lambda_lagrange.size(0) * model_real.lambda_lagrange.size(1))

            leg[epoch] = {
                'W0': model_real.W[0].detach().clone(),
                'W1': model_real.W[1].detach().clone(),
                'z0': model_real.z[0].detach().clone(),
                'a0': model_real.a[0].detach().clone(),
                'z1': model_real.z[1].detach().clone(),
                'lam': model_real.lambda_lagrange.detach().clone(),
                'acc': acc_leg, 'mse': mse_leg, 'lagr': lagr_leg, 'primal': primal_leg,
                'pre': pre_leg, 'act': act_leg, 'firing_rates': firing_rates_leg, 'lam_sum': lam_sum_leg
            }

    end_time_leg = time.time()
    time_leg = end_time_leg - start_time_leg
    
    # ==========================================
    # COMPARISON
    # ==========================================
    print("\n" + "="*55)
    print("  OLD SCRIPT COMPARATION TEST")
    print("="*55)
    for epoch in sorted(oop.keys()):
        diff_W0 = torch.max(torch.abs(oop[epoch]['W0'] - leg[epoch]['W0'])).item()
        diff_W1 = torch.max(torch.abs(oop[epoch]['W1'] - leg[epoch]['W1'])).item()
        diff_z0 = torch.max(torch.abs(oop[epoch]['z0'] - leg[epoch]['z0'])).item()
        diff_a0 = torch.max(torch.abs(oop[epoch]['a0'] - leg[epoch]['a0'])).item()
        diff_z1 = torch.max(torch.abs(oop[epoch]['z1'] - leg[epoch]['z1'])).item()
        
        diff_lam = torch.max(torch.abs(oop[epoch]['lam'] - leg[epoch]['lam'])).item()
        diff_mse = abs(oop[epoch]['mse'] - leg[epoch]['mse'])
        diff_acc = abs(oop[epoch]['acc'] - leg[epoch]['acc'])
        diff_lam_sum = abs(oop[epoch]['lam_sum'] - leg[epoch]['lam_sum'])
        diff_lagr = abs(oop[epoch]['lagr'] - leg[epoch]['lagr'])
        diff_primal = abs(oop[epoch]['primal'] - leg[epoch]['primal'])
        
        
        diff_fr = max([abs(o - l) for o, l in zip(oop[epoch]['firing_rates'], leg[epoch]['firing_rates'])])
        diff_pre = max([abs(o - l) for o, l in zip(oop[epoch]['pre'], leg[epoch]['pre'])])
        diff_act = max([abs(o - l) for o, l in zip(oop[epoch]['act'], leg[epoch]['act'])])
        print("\n")
        print(f"Epoch [{epoch:3d}/{epochs}]")
        print(f"[W0]    Difference: {diff_W0:.8e} " + ("✅" if diff_W0< 1e-3 else "❌"))
        print(f"[W1]    Difference: {diff_W1:.8e} " + ("✅" if diff_W1 < 1e-3 else "❌"))
        print(f"[z0]    Difference: {diff_z0:.8e} " + ("✅" if diff_z0< 1e-3 else "❌"))
        print(f"[z1]    Difference: {diff_z1:.8e} " + ("✅" if diff_z1 < 1e-3 else "❌"))
        print(f"[a0]    Difference: {diff_a0:.8e} " + ("✅" if diff_a0< 1e-3 else "❌"))
        print(f"[W1]    Difference: {diff_W1:.8e} " + ("✅" if diff_W1 < 1e-3 else "❌"))
        
        print("\n")
        print(f"[Metrics] Difference Check:")
        print(f"| MSE:   {diff_mse:.2e} " + ("✅" if diff_mse < 1e-4 else "❌") + 
              f" | Acc:   {diff_acc:.2e} " + ("✅" if diff_acc < 1e-2 else "❌") + 
              f" | LamSum: {diff_lam_sum:.2e} " + ("✅" if diff_lam_sum < 1e-6 else "❌"))
        print(f"| Lagr:  {diff_lagr:.2e} " + ("✅" if diff_lagr < 1e-1 else "❌") + 
              f" | Lamb:  {diff_primal:.2e} " + ("✅" if diff_primal < 1e-2 else "❌") + 
              f" | F-Rate: {diff_fr:.2e} " + ("✅" if diff_fr < 1e-3 else "❌"))
        print(f"| Pre:   {diff_pre:.2e} " + ("✅" if diff_pre < 1e-3 else "❌") + 
              f" | Act:   {diff_act:.2e} " + ("✅" if diff_act < 1e-3 else "❌"))
        print("\n")
        
        # print(oop[epoch]['acc'], leg[epoch]['acc'])
        print("-" * 65)

    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"OOP Total Time:    {time_oop:.2f} seconds")
    print(f"Legacy Total Time: {time_leg:.2f} seconds")
    if time_oop < time_leg:
        print(f"-> The optimized OOP model is {time_leg / time_oop:.2f}x FASTER than the legacy model!")
    else:
        print(f"-> The legacy model is {time_oop / time_leg:.2f}x faster than the OOP model.")