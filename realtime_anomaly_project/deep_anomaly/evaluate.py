import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, roc_curve, precision_recall_curve

from realtime_anomaly_project.deep_anomaly.transformer_ae import TAEGAN


def generate_sine_batch(n_samples: int, seq_len: int, n_features: int, noise_std: float = 0.05):
    """Generate a batch of multivariate sine-like sequences with small noise."""
    t = np.linspace(0, 2 * np.pi, seq_len)
    X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    for i in range(n_samples):
        for f in range(n_features):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.5, 1.5)
            X[i, :, f] = amp * np.sin(freq * t + phase)
        X[i] += np.random.normal(scale=noise_std, size=(seq_len, n_features))
    return X


def inject_anomalies(X: np.ndarray, fraction: float = 0.1, magnitude: float = 3.0, max_length: int = 10):
    """Inject contiguous anomaly windows into a fraction of the samples.
    Returns (X_mod, labels) where labels is 0/1 per sample.
    """
    X_mod = X.copy()
    n = X.shape[0]
    n_anom = max(1, int(n * fraction))
    anom_idx = np.random.choice(n, n_anom, replace=False)
    labels = np.zeros(n, dtype=int)
    for i in anom_idx:
        labels[i] = 1
        length = np.random.randint(1, max_length + 1)
        start = np.random.randint(0, X.shape[1] - length + 1)
        # add a large spike or ramp
        for f in range(X.shape[2]):
            X_mod[i, start:start+length, f] += magnitude * (np.random.rand(length) - 0.5)
    return X_mod, labels


def compute_recon_scores(model: TAEGAN, X: np.ndarray, device: str = 'cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        recon = model.reconstruct(X_t).cpu().numpy()
    # per-sample MSE over time and features
    scores = ((recon - X) ** 2).mean(axis=(1, 2))
    return scores


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int):
    # higher scores are more anomalous
    idx = np.argsort(scores)[::-1]
    topk = idx[:k]
    if len(topk) == 0:
        return 0.0
    return float(y_true[topk].sum()) / len(topk)


def plot_roc_pr(y_true: np.ndarray, scores: np.ndarray, out_dir: str):
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(y_true, scores):.3f}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(out_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()

    plt.figure(figsize=(6, 5))
    ap = average_precision_score(y_true, scores)
    plt.plot(rec, prec, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(out_dir, 'pr_curve.png')
    plt.savefig(pr_path)
    plt.close()
    return roc_path, pr_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=200, help='Number of synthetic samples')
    p.add_argument('--seq', type=int, default=128, help='Sequence length')
    p.add_argument('--feat', type=int, default=3, help='Number of features')
    p.add_argument('--anom_frac', type=float, default=0.15, help='Fraction of samples to make anomalous')
    p.add_argument('--magnitude', type=float, default=4.0, help='Anomaly magnitude')
    p.add_argument('--k', type=int, default=20, help='k for precision@k')
    p.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (.pt/.pth) to load weights from')
    p.add_argument('--plot', action='store_true', help='Save ROC/PR plots to out_dir')
    p.add_argument('--out_dir', type=str, default='.', help='Output directory for plots')
    args = p.parse_args()

    X = generate_sine_batch(args.n, args.seq, args.feat)
    X_mod, labels = inject_anomalies(X, fraction=args.anom_frac, magnitude=args.magnitude)

    model = TAEGAN(input_dim=args.feat)
    # optionally load checkpoint
    if args.checkpoint:
        try:
            ckpt = torch.load(args.checkpoint, map_location=args.device)
            # common checkpoint layouts: dict with 'generator'/'discriminator', 'state_dict', or raw state_dict
            if isinstance(ckpt, dict):
                # try several key patterns
                if 'generator' in ckpt and 'discriminator' in ckpt:
                    model.generator.load_state_dict(ckpt['generator'])
                    model.discriminator.load_state_dict(ckpt['discriminator'])
                elif 'generator_state_dict' in ckpt or 'generator_state' in ckpt:
                    gkey = 'generator_state_dict' if 'generator_state_dict' in ckpt else 'generator_state'
                    model.generator.load_state_dict(ckpt[gkey])
                    if 'discriminator_state_dict' in ckpt:
                        model.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
                elif 'state_dict' in ckpt:
                    try:
                        model.load_state_dict(ckpt['state_dict'], strict=False)
                    except Exception:
                        try:
                            model.generator.load_state_dict(ckpt['state_dict'], strict=False)
                        except Exception:
                            pass
                else:
                    try:
                        model.load_state_dict(ckpt, strict=False)
                    except Exception:
                        try:
                            model.generator.load_state_dict(ckpt, strict=False)
                        except Exception:
                            print('Warning: checkpoint provided but could not be loaded cleanly; proceeding with random init')
            else:
                print('Warning: unsupported checkpoint format; ignoring')
        except Exception as e:
            print('Failed to load checkpoint', args.checkpoint, e)

    scores = compute_recon_scores(model, X_mod, device=args.device)

    # metrics
    try:
        roc_auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
    except Exception as e:
        print('Error computing AUC/AP:', e)
        roc_auc = float('nan')
        ap = float('nan')

    preck = precision_at_k(labels, scores, args.k)

    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'Average Precision (AP): {ap:.4f}')
    print(f'Precision@{args.k}: {preck:.4f}')

    if args.plot:
        os.makedirs(args.out_dir, exist_ok=True)
        roc_path, pr_path = plot_roc_pr(labels, scores, args.out_dir)
        print('Saved ROC to', roc_path)
        print('Saved PR to', pr_path)


if __name__ == '__main__':
    main()
