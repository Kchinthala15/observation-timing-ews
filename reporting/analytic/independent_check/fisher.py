import numpy as np
rng=np.random.default_rng(7)
def neff_formula(n,s):
    k=np.arange(1,n); return n*n/(n+2*np.sum((n-k)*s**k))
print("finite-n formula 0.6,0.2 even:",neff_formula(60,np.exp(-2*0.2/0.6)))
def crlb_V(t,V,th,known_theta):
    D=np.abs(t[:,None]-t[None,:]); R=np.exp(-th*D)
    S=V*R; Si=np.linalg.inv(S)
    dV=R; dT=-V*D*R
    A=Si@dV; B=Si@dT
    I=np.array([[np.trace(A@A),np.trace(A@B)],[np.trace(B@A),np.trace(B@B)]])/2
    if known_theta: return 1/I[0,0]
    return np.linalg.inv(I)[0,0]
n=60;V=1.0
for lam,th in [(0.6,1.2),(0.6,0.2),(0.3,0.2)]:
    tE=np.arange(n)/lam
    cE=crlb_V(tE,V,th,False)
    cP=np.mean([crlb_V(np.concatenate([[0],np.cumsum(rng.exponential(1/lam,n-1))]),V,th,False) for _ in range(400)])
    print(lam,th,"CRLB n_eff (theta unknown, times known): even %.1f  poisson %.1f ; theta known: %.1f"%(2/cE,2/cP,2/crlb_V(tE,V,th,True)))
# compression crossover
from scipy.optimize import brentq
f=lambda x:(1+x)**2-np.exp(x)
print("slope crossover x=",brentq(f,1,5))
lam=0.6
for th1,th2 in [(1.2,0.05),(3*lam,lam),(5*lam,3*lam)]:
    print(th1,th2,"even",np.exp(-th2/lam)-np.exp(-th1/lam),"pois",lam/(lam+th2)-lam/(lam+th1))
