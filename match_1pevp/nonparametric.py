import numpy as np
from numpy.linalg import eigvals, svd, solve
from scipy.sparse.linalg import factorized

def solveLS(A, b):
    if isinstance(A, np.ndarray): return solve(A, b)
    return factorized(A)(b)

def beyn(L, center, radius, lhs, rhs, N_quad, rank_tol, hankel = 1):
    """    
    This function computes approximate eigenvalues of non-parametric eigenproblems through Beyn's contour integral method
    
    Parameters:
    L: lambda function defining matrix in eigenproblem
    center: center of contour (disk)
    radius: radius of contour (disk)
    lhs: left-sketching matrix
    rhs: right-sketching matrix
    N_quad: number of quadrature points
    rank_tol: tolerance for rank truncation
    hankel: size of block-Hankel matrices
    
    Returns:
    vals: approximate eigenvalues
    """
    ts = center + radius * np.exp(1j * np.linspace(0., 2 * np.pi, N_quad + 1)[: -1])
    res_flat = np.array([(lhs @ solveLS(L(t), rhs)).reshape(-1) for t in ts])
    dft = ts.reshape(-1, 1) ** (1 + np.arange(2 * hankel))
    quad = dft.T * (ts - center)
    As_flat = quad @ res_flat
    As = As_flat.reshape(2 * hankel, lhs.shape[0], rhs.shape[1])
    H0 = np.block([[As[i + j] for j in range(hankel)] for i in range(hankel)])
    H1 = np.block([[As[i + j + 1] for j in range(hankel)] for i in range(hankel)])
    u, s, vh = svd(H0)
    r_eff = np.where(s > rank_tol * s[0])[0][-1] + 1
    u, s, vh = u[:, : r_eff], s[: r_eff], vh[: r_eff, :]
    B = u.T.conj() @ H1 @ (vh.T.conj() / s[..., None, :])
    vals = eigvals(B)
    vals = vals[abs(vals - center) <= radius]
    return vals


def loewner(L, center, radius, lhs, rhs, lint, rint, N_quad, rank_tol):
        '''
        Parameters:
            L: lambda function defining matrix in eigenproblem
            center: center of contour (disk)
            radius: radius of contour (disk)
            lhs: left-sketching matrix
            rhs: right-sketching matrix
            lint: left interpolation points
            rint: right interpolation points
            N_quad: number of quadrature points
            rank_tol: tolerance for rank truncation
    
        Returns:
            vals : approximate eigenvalues
        '''
        
        # currently I do not know what I am doing so take the following code with a grain of salt
        ts = center + radius * np.exp(1j * np.linspace(0., 2 * np.pi, N_quad + 1)[: -1])
        QR = np.array([(solveLS(L(t), rhs)) for t in ts])
        QL = np.array([(solveLS(L(t).T.conj(), lhs.T.conj())).T.conj() for t in ts])
        dft_l = np.array([(1 / (lint[i] - ts)) for i in range(len(lint))]) 
        dft_r = np.array([(1 / (rint[i] - ts)) for i in range(len(rint))])
        quad_l = dft_l * (ts - center) #* (1/ N_quad) # left weights
        quad_r = dft_r * (ts - center) #* (1 / N_quad) # right weights
        cauchy = 1.0 / (lint.reshape((-1,1)) - rint)
        H_eval_l = np.array([quad_l[i,:] @ QL[:,i,:] for i in range(len(lint))]) 
        H_eval_r = np.array([quad_r[i,:] @ QR[:,:,i] for i in range(len(rint))]) 

        # Loewner matrices 
        Lo = cauchy * (H_eval_l @ rhs - lhs @ H_eval_r.T)  # see eq.18 in https://doi.org/10.1007/s10915-022-01800-3
        So = cauchy * (np.diag(lint) @ H_eval_l @ rhs - lhs @ H_eval_r.T @ np.diag(rint))

        u, s, vh = svd(Lo)
        r_eff = np.where(s > 1e-10 * s[0])[0][-1] + 1
        u, s, vh = u[:, : r_eff], s[: r_eff], vh[: r_eff, :]
        B = np.diag(1/s) @ u.T.conj() @ So @ vh.T.conj() 
        vals = eigvals(B)
        vals = vals[abs(vals - center) <= radius]
                
        return vals
