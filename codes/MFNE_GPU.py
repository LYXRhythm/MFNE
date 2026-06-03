import numpy as np
import torch
from GB_generation_with_idx import get_GB
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


# =========================================================
# GPU-accelerated MFNE
# =========================================================
def MFNE_torch(X, sigma, device=None, dtype=torch.float32, eps=1e-8):
    """
    PyTorch GPU version of MFNE.

    Parameters
    ----------
    X : ndarray, shape (n_granular_balls, n_attributes)
        Granular-ball centers.

    sigma : float
        Fuzzy neighborhood radius parameter.

    device : str or torch.device
        'cuda' or 'cpu'. If None, use CUDA if available.

    dtype : torch.dtype
        Default is torch.float32 for GPU efficiency.

    eps : float
        Small value for numerical stability.

    Returns
    -------
    OF : ndarray, shape (n_granular_balls,)
        Granular-ball-level outlier scores.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    X = torch.as_tensor(X, dtype=dtype, device=device)

    n, m = X.shape

    # Shape: (m,)
    std = torch.std(X, dim=0, unbiased=False)
    std_safe = torch.clamp(std, min=eps)

    # Shape: (m,)
    radius = std_safe / sigma

    # Shape: (n, m) -> (m, n)
    X_attr = X.T

    # Shape: (m, n, n)
    diff = torch.abs(X_attr[:, :, None] - X_attr[:, None, :])

    # Attribute-wise Mahalanobis-like distance
    dist = diff / std_safe[:, None, None]

    # Shape: (m, n, n)
    R = 1.0 - dist

    # Threshold by radius
    R = torch.where(R >= radius[:, None, None], R, torch.zeros_like(R))

    # Avoid negative fuzzy relation values caused by large distances
    R = torch.clamp(R, min=0.0)

    # Shape: (m, n)
    row_sum = torch.sum(R, dim=2)

    # Shape: (m,)
    FNE = -torch.sum(
        torch.log2(torch.clamp(row_sum / n, min=eps)),
        dim=1
    ) / n

    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    mask = mask[None, :, :]  # Shape: (1, n, n)

    log_terms = torch.log2(
        torch.clamp(adjusted_row_sum / (n - 1), min=eps)
    )

    log_terms = torch.where(mask, log_terms, torch.zeros_like(log_terms))

    # Shape: (m, n), indexed as remove_x_E_attr[k, i]
    remove_x_E_attr = -torch.sum(log_terms, dim=1) / (n - 1)

    # Shape: (n, m)
    remove_x_E = remove_x_E_attr.T

    # Shape: (m, n)
    col_sum = torch.sum(R, dim=1)

    # Shape: (m,)
    total_sum = torch.sum(R, dim=(1, 2))

    # Shape: (m, n)
    diag_R = torch.diagonal(R, dim1=1, dim2=2)

    # Shape: (m, n)
    temp_x_sum = total_sum[:, None] - row_sum - col_sum + diag_R

    # Shape: (m, n)
    rnc_attr = row_sum - temp_x_sum / (n - 1)

    # Shape: (n, m)
    rnc = rnc_attr.T

    FNE_safe = torch.clamp(FNE, min=eps)

    # Shape: (n, m)
    rne = 1.0 - remove_x_E / FNE_safe[None, :]

    rne = torch.where((rne >= 0.0) & (rne <= 1.0), rne, torch.zeros_like(rne))

    abs_rnc = torch.abs(rnc)

    nod_pos = rne * (n - abs_rnc) / (2.0 * n)
    nod_neg = rne * torch.sqrt(torch.clamp((n + abs_rnc) / (2.0 * n), min=0.0))

    nod = torch.where(rnc > 0.0, nod_pos, nod_neg)
    OF = 1.0 - torch.sum((1.0 - nod) * weight, dim=1) / m

    OF = torch.nan_to_num(OF, nan=0.0, posinf=0.0, neginf=0.0)

    return OF.detach().cpu().numpy()
