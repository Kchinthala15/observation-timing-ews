import numpy as np
rng=np.random.default_rng(12345)
def neff_formula(n,s):
    k=np.arange(1,n); return n*n/(n+2*np.sum((n-k)*s**k))
def sim(lam,th,n,R,design):
    if design=='E': gaps=np.full((R,n-1),1/lam)
    else: gaps=rng.exponential(1/lam,(R,n-1))
    a=np.exp(-th*gaps)
    X=np.empty((R,n)); X[:,0]=rng.standard_normal(R)
    for i in range(n-1):
        X[:,i+1]=a[:,i]*X[:,i]+np.sqrt(1-a[:,i]**2)*rng.standard_normal(R)
    return X
n=60;R=40000
print("lam th | rhoE mc/th | rhoP mc/th | sampleACF E / P | neffE mc/f/largen | neffP mc/f/largen | AR1-twin neff (s=rhoP^2)")
for lam,th in [(0.6,1.2),(0.6,0.2),(0.3,0.2),(0.6,0.05),(1.2,0.2)]:
    out=[]
    res={}
    for d in 'EP':
        X=sim(lam,th,n,R,d)
        prod=np.mean(X[:,:-1]*X[:,1:])
        Xc=X-X.mean(1,keepdims=True)
        acf=np.sum(Xc[:,:-1]*Xc[:,1:],1)/np.sum(Xc**2,1)
        S=np.mean(X**2,1)
        res[d]=(prod,acf.mean(),2/np.var(S))
    rE=np.exp(-th/lam); rP=lam/(lam+th); sE=np.exp(-2*th/lam); sP=lam/(lam+2*th)
    print(f"{lam} {th} | {res['E'][0]:.3f}/{rE:.3f} | {res['P'][0]:.3f}/{rP:.3f} | {res['E'][1]:.3f}/{res['P'][1]:.3f} | {res['E'][2]:.1f}/{neff_formula(n,sE):.1f}/{n*(1-sE)/(1+sE):.1f} | {res['P'][2]:.1f}/{neff_formula(n,sP):.1f}/{n*(1-sP)/(1+sP):.1f} | {neff_formula(n,rP**2):.1f}")
# long series: sample ACF convergence under Poisson
for lam,th in [(0.6,1.2),(0.3,0.2)]:
    X=sim(lam,th,200000,1,'P')[0]; Xc=X-X.mean()
    print("long-run sample ACF",lam,th,np.sum(Xc[:-1]*Xc[1:])/np.sum(Xc**2), lam/(lam+th))
