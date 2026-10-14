"""Run the Figure 1 supervised disentanglement simulation locally.

The simulation samples class-conditional latent variables, passes them through
a fixed invertible nonlinear mixing network, and trains an encoder plus linear
class readout. Recovery metrics and correlations are saved periodically under
``Data/fig1/runs``. Shared numerical primitives are defined in ``Codes/utils.py``.
"""

import argparse
import csv
import json
import os
import random
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODES_DIR))

from utils import (  # noqa: E402
    NSphereSpace,
    construct_invertible_mlp,
    disentanglement_score,
    get_mlp,
    linear_disentanglement,
)


DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Disentanglement with DIET - MLP Mixing"
    )

    # General

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--nr-indices", type=int, default=10000,
                        help="Number of indices and, therefore, number of rows of matrix W.")

    # Latent distribution

    parser.add_argument(
        "--n", type=int, default=5, help="Dimensionality of the latents and observed variables. "
        "Potential values to start with: 5, 10."
    )
    parser.add_argument(
        "--nh", type=int, default=0, help="Dimensionality of the learned latents."
        "If 0, set to equal n."
    )

    parser.add_argument(
        "--nr-clusters", type=int, default=100, help="Number of cluster vectors.")

    parser.add_argument(
        "--m-p",
        type=int,
        default=0,
        help="Type of distribution of cluster vectors (all coming from the sphere). p=0 means uniform on the sphere, "
        "all other p values correspond to (projected) L_p Exponential around the first unit vector",
    )

    parser.add_argument(
        "--m-param",
        type=float,
        default=1.0,
        help="Additional parameter for the marginal (only relevant if it is not uniform) to quantify the concentration.",
    )
    parser.add_argument(
        "--c-p",
        type=int,
        default=0,
        help="Type of distribution of cluster vectors (all coming from the sphere). p=0 means uniform on the sphere, "
        "all other p values correspond to (projected) L_p Exponential around the first unit vector",
    )

    parser.add_argument(
        "--c-param",
        type=float,
        default=10,
        help="Concentration parameter of the vMF conditional distribution around cluster vectors."
        "Working values start around 5 and can go up to high double digits.",
    )

    # Mixing network

    parser.add_argument(
        "--n-mixing-layer",
        type=int,
        default=4,
        help="Number of layers in nonlinear mixing network g.",
    )

    # Encoder

    parser.add_argument(
        "--n-hidden-layers",
        type=int,
        default=6,
        help="Number of hidden layers in the encoder. Minimum 2.",
    )

    parser.add_argument(
        "--sphere-norm", action="store_true", help="Normalize output to a sphere."
    )

    parser.add_argument(
        "--no-batch-norm", action="store_true", help="Do not use batch normalization in the encoder.")


    # Training

    parser.add_argument("--W-bias-used", action="store_true", help="Use bias in W matrix.")
    parser.add_argument("--normalize-rows-W", action="store_true", help="Normalize rows of W matrix.")


    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--n-log-steps", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument(
        "--grad_clip", type=int, default=0, help="Clip gradients to norm 1."
    )
    parser.add_argument("--mcc_stop", type=float, default=2, 
                        help="stop training when mcc hits this.")

    # Evaluation

    parser.add_argument(
        "--mode", type=str, default="pearson",
        help="Metric: r2, adjusted_r2, pearson, spearman, mae"
    )

    # Output

    parser.add_argument(
        "--save_folder", type=str, default=str(REPO_ROOT / "Data" / "fig1" / "runs"),
        help="Root directory for one subdirectory per simulation configuration."
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Also save model and data-generating-process state dictionaries.",
    )

    # Exponential latent model
    parser.add_argument(
        "--expon", type=int, default=0, help="Not a sphere, exponential distribution."
    )
    parser.add_argument(
        "--expon_enc", type=int, default=0, help='whether to use exponential encoder.'
    )
    parser.add_argument(
        "--relu_enc", type=int, default=0, help='whether to use relu encoder.'
    )
    parser.add_argument(
        "--softplus", type=int, default=0, help='whether to use softplus encoder.'
    )
    parser.add_argument(
        "--enc_bn", type=int, default=0, help='for relu encoder, use BN before relu if true.'
    )
    parser.add_argument(
        "--expon_entropy",
        type=float,
        default=1,
        help="Entropy of cluster vectors (lower values -> more sparse). "
        "If expon_entropy is positive, just use initialized mean.",
    )
    parser.add_argument(
        "--expon_scale",
        type=float,
        default=.1,
        help="Scale of exponential distribution from cluster vectors; larger -> z has less variance.",
    )
    parser.add_argument(
        "--aniso", type=int, default=0, help="More variance in first component."
    )
    parser.add_argument(
        "--aniso_fac",
        type=float,
        default=10,
        help="Norm of optimal W for aniso component will be that much higher.",
    )


    args = parser.parse_args()
    
    if args.nh == 0:
        args.nh = args.n

    assert args.expon_enc + args.relu_enc + args.softplus < 2

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    return args


def create_generative_process(args):
    if args.expon:
        def sample_cluster_vectors(args):
            """Sample positive exponential rates with same entropy across clusters."""
            if type(args.seed) != type(None):
                torch.manual_seed(args.seed)
            logits = torch.randn(args.nr_clusters, args.n, requires_grad=True)
            optim = torch.optim.Adam([logits], lr=1e-3)
            if args.expon_entropy > 0:
                # Match the entropy implied by the initialization.
                cluster_vectors = torch.softmax(logits, dim=1)
                h = torch.sum(torch.log(cluster_vectors), dim=1)
                expon_entropy = h.mean().detach()
            else:
                expon_entropy = args.expon_entropy
            for i in range(20000):
                cluster_vectors = torch.nn.functional.softplus(logits)
                h = torch.sum(torch.log(cluster_vectors), dim=1)
                loss = torch.mean((h - expon_entropy)**2)
                
                # Optionally increase the first coordinate norm.
                if args.aniso:
                    norms = torch.linalg.norm(cluster_vectors, dim=0)
                    tmp = torch.ones_like(norms)                
                    tmp[0] *= args.aniso_fac
                    loss = loss + torch.mean((norms - tmp)**2)
                optim.zero_grad()
                loss.backward()
                optim.step()
            print('Cluster Entropy=%.4e, loss=%.4e, std=%.4e' % (expon_entropy, loss.item(), h.std().item()))
            return cluster_vectors
        
        cluster_vectors = sample_cluster_vectors(args).to(DEVICE)
    
        def sample_conditional(z, device=DEVICE):
            p = torch.distributions.Exponential(
                rate=args.expon_scale/z
            )
            return p.sample()
        
        def sample_latents(size, cluster_vectors=cluster_vectors, device=DEVICE):
            indices = torch.randint(0, args.nr_indices, (size,), device=device)
            cluster_indices = indices % args.nr_clusters
            corresponding_cluster_vectors = cluster_vectors[cluster_indices]
            samples = sample_conditional(corresponding_cluster_vectors, device=device)

            return samples, indices, cluster_indices
        
    else:
        space = NSphereSpace(args.n)

        eta = torch.zeros(args.n)
        eta[0] = 1.0

        if args.m_p != 0:
            if args.m_p == 1:
                sample_marginal = lambda space, size, device=DEVICE: space.laplace(
                    eta, 1.0, size, device
                )
            elif args.m_p == 2:
                sample_marginal = lambda space, size, device=DEVICE: space.normal(
                    eta, 1.0, size, device
                )
            else:
                sample_marginal = (
                    lambda space, size, device=DEVICE: space.generalized_normal(
                        eta, args.m_param, p=args.m_p, size=size, device=device
                    )
                )
        else:
            sample_marginal = lambda space, size, device=DEVICE: space.uniform(
                size, device=device
            )

        if args.c_p != 0:
            if args.c_p == 1:
                sample_conditional = (
                    lambda space, z, size, device=DEVICE: space.laplace(
                    z, 1.0, size, device)
                )
            elif args.c_p == 2:
                sample_conditional = (
                    lambda space, z, size, device=DEVICE: space.normal(
                    z, 1.0, size, device)
                )
            else:
                raise NotImplementedError
        else:
            sample_conditional = (
                lambda space, z, size, device=DEVICE: space.von_mises_fisher(
                z, args.c_param, size, device)
            )

        sample_cluster_vectors = (
            lambda nr_clusters, device=DEVICE: sample_marginal(space, size=nr_clusters, device=device)
        )
        
        cluster_vectors = sample_cluster_vectors(args.nr_clusters)
        
        def sample_latents(size, cluster_vectors=cluster_vectors, device=DEVICE):
            indices = torch.randint(0, args.nr_indices, (size,), device=device)
            cluster_indices = indices % args.nr_clusters
            corresponding_cluster_vectors = cluster_vectors[cluster_indices]
            samples = sample_conditional(space, corresponding_cluster_vectors, size, device=device)

            return samples, indices, cluster_indices
    
    g = construct_invertible_mlp(
        n=args.n,
        n_layers=args.n_mixing_layer,
        act_fct="leaky_relu",
        cond_thresh_ratio=0.0,
        n_iter_cond_thresh=25000,
    )
    g = g.to(DEVICE)

    for p in g.parameters():
        p.requires_grad = False

    return cluster_vectors, sample_latents, g



class BatchNormScaleOnly(torch.nn.BatchNorm1d):
    def __init__(self, num_features, affine=False, **kwargs):
        super().__init__(num_features, affine=affine, **kwargs)

    def forward(self, x):
        # Apply scale without a learned bias.
        return torch.nn.functional.batch_norm(
            x, self.running_mean * 0, self.running_var, 
            self.weight, None, self.training, self.momentum, self.eps
        )
    
class ExpEncoder(torch.nn.Module):
    def __init__(self, f, affine=False):
        super().__init__()
        self.f = f
        self.bn = BatchNormScaleOnly(f[-1].out_features, affine=affine)
        self.act = torch.exp
    
    def forward(self, x):
        z = self.f(x)
        z = self.bn(z)
        z = self.act(z)
        return z
    
class ReLUEncoder(torch.nn.Module):
    def __init__(self, f, enc_bn=False):
        super().__init__()
        self.f = f
        self.enc_bn = enc_bn
        if self.enc_bn:
            self.bn = BatchNormScaleOnly(f[-1].out_features)
        self.act = torch.nn.ReLU()
    
    def forward(self, x):
        z = self.f(x)
        if self.enc_bn:
            z = self.bn(z)
        z = self.act(z)
        return z
    
class SoftplusEncoder(torch.nn.Module):
    def __init__(self, f, enc_bn=False):
        super().__init__()
        self.f = f
        self.enc_bn = enc_bn
        if self.enc_bn:
            self.bn = BatchNormScaleOnly(f[-1].out_features)
        self.act = torch.nn.Softplus()
    
    def forward(self, x):
        z = self.f(x)
        if self.enc_bn:
            z = self.bn(z)
        z = self.act(z)
        return z
    

def create_encoder(args):
    if args.sphere_norm:
        output_normalization = "fixed_sphere"
    else:
        output_normalization = None

    if args.n_hidden_layers < 2:
        raise ValueError("n_hidden_layers < 2: Encoder must have at least 2 hidden layers")

    layers = [args.n * 10] + [args.n * 50] * (args.n_hidden_layers - 2) + [args.n * 10]
    
    encoder = get_mlp(
        n_in=args.n,
        n_out=args.nh,
        layers=layers,
        layer_normalization="bn" if not args.no_batch_norm else None,
        output_normalization=output_normalization,
    ).to(DEVICE)

    if args.expon_enc and args.relu_enc:
        raise ValueError("Cannot use both exp and relu enc.")
    
    if args.expon_enc:
        encoder = ExpEncoder(encoder).to(DEVICE)
    elif args.relu_enc:
        encoder = ReLUEncoder(encoder, enc_bn=args.enc_bn).to(DEVICE)
    elif args.softplus:
        encoder = SoftplusEncoder(encoder, enc_bn=args.enc_bn).to(DEVICE)

    return encoder


def torch_normalize(a):
    a_ = a - torch.mean(a, dim=0, keepdim=True)
    norm = torch.linalg.norm(a_, dim=0, keepdim=True)
    norm = torch.where(norm == 0, torch.tensor(1.0, device=norm.device), norm)  # avoid div by zero
    return a_ / norm


def torch_corr(a, b):
    return torch_normalize(a).T @ torch_normalize(b)
    
    
def test_disentanglement(args, sample_function, cluster_vectors, h, h2=None, n_samples=4096, return_linear_model=False, mode="r2"):
    _h2 = h2 if h2 is not None else lambda z: z
    z_disentanglement, _, _ = sample_function(n_samples,cluster_vectors)
    z = _h2(z_disentanglement)
    hz = h(z_disentanglement)
    corr = torch_corr(z, hz).detach().cpu().numpy()
    return_tuple = linear_disentanglement(
        z, hz, mode=mode, train_test_split=True,
    )
    (hz_to_z, _), _ = return_tuple
    return_tuple = linear_disentanglement(
        hz, z, mode=mode, train_test_split=True,
    )
    (z_to_hz, _), _ = return_tuple
    if args.n == args.nh:
        mcc, mcc_nl = disentanglement_score(
            z, hz, mode=mode, reorder=True, nonlinear=True
        )
    else:
        mcc = mcc_nl = 0
    return hz_to_z, z_to_hz, mcc, mcc_nl, corr


def save_log(args, hz_to_zs, z_to_hzs, mccs, mcc_nls, losses):
    with open(os.path.join(args.save_folder, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'step', 'seed', 'nb indices', 'batch size', 'nb clusters', 'n', 
            'm', 'm param', 'c', 'c param', 'z norm', 'w bias', 'w norm', 
            'loss', 'epochs', 'hz_to_z', 'z_to_hz', 'mcc', 'mcc_nl'
        ])
        for ((step, hz_to_z), (_, z_to_hz), (_, mcc), (_, mcc_nl), (_, loss)) in zip(hz_to_zs, z_to_hzs, mccs, mcc_nls, losses):
            writer.writerow([
                step, args.seed, args.nr_indices, args.batch_size, 
                args.nr_clusters, args.n, args.m_p, args.m_param, 
                args.c_p, args.c_param, args.sphere_norm, 
                args.W_bias_used, args.normalize_rows_W, 
                loss, args.n_epochs, hz_to_z, z_to_hz, mcc, mcc_nl
            ])


def save_corr(args, corrs):
    """
    Parameters
    ----------
    args   : argparse.Namespace
             Must contain `save_folder`.
    corrs  : list[(int, np.ndarray or torch.Tensor)]
             Each entry is (step, correlation_matrix) with
             identical shapes across the list.
    """

    if not corrs:
        raise ValueError("corrs list is empty")

    first_mat = corrs[0][1]
    if torch.is_tensor(first_mat):
        first_mat = first_mat.cpu().numpy()
    shape = first_mat.shape
    if len(shape) != 2:
        raise ValueError(f"corr matrices must be 2-D, got shape {shape}")
    r, c = shape

    header = ['step'] + [f'{i}_{j}' for i in range(r) for j in range(c)]

    path = os.path.join(args.save_folder, 'corr.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for step, mat in corrs:
            if torch.is_tensor(mat):
                mat = mat.cpu().numpy()
            if mat.shape != shape:
                raise ValueError(
                    f"Inconsistent corr shape at step {step}: "
                    f"expected {shape}, got {mat.shape}"
                )
            writer.writerow([step] + mat.reshape(-1).tolist())
                    

def train_diet(args, cluster_vectors, sample_latents, g, f, logger, sparsity_weight=0):
    n_samples = 10000
    z_test, _, _ = sample_latents(n_samples, cluster_vectors)
    
    W = torch.nn.Linear(args.nh, args.nr_clusters, bias=args.W_bias_used, device=DEVICE)

    optimizer_f = torch.optim.AdamW(
        list(f.parameters())+list(W.parameters()), 
        lr=args.lr, weight_decay=args.weight_decay
    )

    losses = []
    hz_to_zs = []
    z_to_hzs = []
    mccs = []
    mcc_nls = []
    corrs = []

    step = 0
    stop = False
    for _ in range(args.n_epochs):
        f.train().to(DEVICE)
        W.train().to(DEVICE)
        run_loss = []
        for _ in range(args.nr_indices//args.batch_size):
            step += 1
            zgt, _, indices = sample_latents(args.batch_size)
            if args.normalize_rows_W:
                W.weight.data = W.weight.data/torch.norm(W.weight.data,p=2,dim=-1,keepdim=True)
            h = f(g(zgt))
            zinf = W(h)

            loss = F.cross_entropy(zinf, indices.to(DEVICE))
                
            optimizer_f.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(W.parameters(), max_norm=1.0)
            optimizer_f.step()

            run_loss.append(loss.item())
            if step % args.n_log_steps == 0 or step == (args.n_epochs*(args.nr_indices//args.batch_size)) - 1:
                f.eval()
                hz_to_z, z_to_hz, mcc, mcc_nl, corr = test_disentanglement(
                    args, sample_function=sample_latents, cluster_vectors=cluster_vectors, 
                    h=lambda z: f(g(z)), mode=args.mode)
                hz_to_zs.append((step, hz_to_z))
                z_to_hzs.append((step, z_to_hz))
                mccs.append((step, mcc))
                mcc_nls.append((step, mcc_nl))
                corrs.append((step, corr.copy()))

                f.train()
                metrics = {
                    'loss': np.mean(run_loss), 
                    'lin. hz_to_z': hz_to_z, 
                    'lin. z_to_hz': z_to_hz, 
                    'mcc': mcc,
                }
                for i, c in enumerate(corr[0]):
                    metrics['corr[0,%s]' % i] = c
                logger.update(step, metrics)
                losses.append((step, np.mean(run_loss)))

                if mcc >= args.mcc_stop:
                    stop = True
                    break

        if len(losses) > 0:
            save_log(args, hz_to_zs, z_to_hzs, mccs, mcc_nls, losses)
            save_corr(args, corrs)
        f.eval()
        with torch.no_grad():
            hz = f(g(z_test.to(DEVICE)))
        torch.save(hz.detach().cpu(), os.path.join(args.save_folder, "hz.pt"))
        if args.save_checkpoints:
            torch.save(f.cpu().state_dict(), os.path.join(args.save_folder, "model.pt"))
            torch.save(W.cpu().state_dict(), os.path.join(args.save_folder, "W.pt"))
        f.to(DEVICE)
        W.to(DEVICE)

        if stop:
            print('MCC above %s, early stopping.' % args.mcc_stop)
            break

    return losses, hz_to_zs, z_to_hzs, mccs, mcc_nls, W


class Logger(ABC):
    @abstractmethod
    def update(self, step, metrics: dict):
        pass


class StdoutLogger(Logger):
    def __init__(self, fixed_msg=None):
        self._fixed_msg = fixed_msg if fixed_msg is not None else ""

    def update(self, step, metrics: dict):
        message = self._fixed_msg + f"Step: {step} \t"
        for key, value in metrics.items():
            if isinstance(value, list):
                value = [f"{v:.4f}" for v in value]
                value = ", ".join(value)
                message += f"{key}: [{value}] \t"
            else:
                message += f"{key}: {value:.4f} \t"
        print(message)

def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    tag = (
        f"supervised_seed_{args.seed}_"
        f"nsamp_{args.nr_indices}_"
        f"n_{args.n}_"
        f"nh_{args.nh}_"
        f"nclusters_{args.nr_clusters}_"
        f"epochs_{args.n_epochs}_"
        f"mp_{args.m_p}_"
        f"mparam_{args.m_param}_"
        f"cp_{args.c_p}_"
        f"cparam_{args.c_param}_"
        f"{args.sphere_norm}_"
        f"{args.normalize_rows_W}_"
        f"expon_{args.expon}_"
        f"enc_{args.expon_enc}_"
        f"relu_{args.relu_enc}"
        f"entropy_{args.expon_entropy}_"
        f"scale_{args.expon_scale}"
    )
    
    args.save_folder = os.path.join(args.save_folder,tag)
    os.makedirs(args.save_folder, exist_ok = True)

    with open(os.path.join(args.save_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Device in use: ", DEVICE)
    cluster_vectors, sample_latents, g = create_generative_process(args)
    print("g: ", g)
    if args.save_checkpoints:
        torch.save(g.cpu().state_dict(), os.path.join(args.save_folder, "generator.pt"))
        torch.save(
            cluster_vectors.detach().cpu(),
            os.path.join(args.save_folder, "cluster_vectors.pt"),
        )
    g.to(DEVICE)
    cluster_vectors.to(DEVICE)

    f = create_encoder(args)

    losses, hz_to_zs, z_to_hzs, mccs, mcc_nls, W = train_diet(
        args, cluster_vectors, sample_latents, g, f, StdoutLogger()
    )
    
if __name__ == "__main__":
    main()
