import numpy as np
rng=np.random.default_rng(20261101)
def ou_at(t,theta,V,rng):
    x=np.empty(len(t)); x[0]=rng.normal(0,np.sqrt(V))
    for i in range(1,len(t)):
        a=np.exp(-theta*(t[i]-t[i-1])); x[i]=a*x[i-1]+rng.normal(0,np.sqrt(V*(1-a*a)))
    return x
res=[]
for lam,theta in [(0.6,1.2),(0.6,0.2),(0.6,0.05),(1.2,0.2),(0.3,0.2)]:
    n=60; reps=20000; V=1.0
    r1E=[];r1P=[];svE=[];svP=[]
    for _ in range(reps):
        tE=np.arange(n)/lam; tP=np.cumsum(rng.exponential(1/lam,n))
        xE=ou_at(tE,theta,V,rng); xP=ou_at(tP,theta,V,rng)
        r1E.append(np.mean(xE[1:]*xE[:-1])/V); r1P.append(np.mean(xP[1:]*xP[:-1])/V)   # known mean/variance: lag-1 product correlation
        svE.append(np.mean(xE**2)); svP.append(np.mean(xP**2))   # variance estimator with known mean 0
    rE,rP=np.exp(-theta/lam),lam/(lam+theta)
    sE,sP=np.exp(-2*theta/lam),lam/(lam+2*theta)
    # Var(mean of x^2) for Gaussian = (2V^2/n^2) sum_ij corr_ij^2 ; n_eff = 2V^2/Var
    neffE_emp=2/np.var(svE); neffP_emp=2/np.var(svP)
    neff=lambda s:n*(1-s)/(1+s)
    # exact finite-n expectation: sum_ij s^|i-j|
    def neff_exact(s):
        k=np.arange(1,n); tot=n+2*np.sum((n-k)*s**k); return n*n/tot
    res.append((lam,theta,np.mean(r1E),rE,np.mean(r1P),rP,neffE_emp,neff_exact(sE),neff(sE),neffP_emp,neff_exact(sP),neff(sP)))
print("lam theta | E[r1] even: MC vs e^{-th/lam} | E[r1] Poisson: MC vs lam/(lam+th) | n_eff(var) even: MC, exact-n, large-n | Poisson: MC, exact-n, large-n")
for r in res: print(" ".join(f"{v:7.3f}" for v in r))
