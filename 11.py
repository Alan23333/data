import numpy as np
import tensorly as tl
import t_tools

tl.set_backend('numpy')

def t_svd(A):
    Af = np.fft.fft(A,axis=2)
    U = np.zeros(tuple([A.shape[0],A.shape[0],A.shape[2]]), dtype=complex)
    S = np.zeros(tuple(A.shape), dtype=complex)
    V = np.zeros(tuple([A.shape[1],A.shape[1],A.shape[2]]), dtype=complex)

    for i3 in range(A.shape[2]):
        uf,sf,vf = np.linalg.svd(Af[:,:,i3])
        sf = np.diag(sf)

        U[:,:,i3] = uf
        S[:, :, i3] = sf
        V[:, :, i3] = vf
    U = np.fft.ifft(U,axis=2)
    S = np.fft.ifft(S,axis=2)
    V = np.fft.ifft(V,axis=2)

    Ahat = t_tools.t_prod(t_tools.t_prod(U,S),V)
    return U,S,V,Ahat
A = tl.tensor(np.random.random((5,5,3)))

U,S,V,Ahat = t_svd(A)