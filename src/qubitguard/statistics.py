"""Binomial uncertainty and paired comparisons on the same shots."""
import math
import numpy as np
from scipy.stats import binomtest, beta

def wilson(k, n, z=1.959963984540054):
    if n <= 0 or k < 0 or k > n: raise ValueError('Invalid binomial counts')
    p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (0. if k==0 else max(0.,mid-half)),(1. if k==n else min(1.,mid+half))

def paired(a,b,truth):
    ea=np.asarray(a)!=truth; eb=np.asarray(b)!=truth
    wins=int(np.sum(~ea & eb)); losses=int(np.sum(ea & ~eb)); n=len(truth)
    delta=(losses-wins)/n
    var=max(0.,(wins+losses)/n-delta**2)/n
    def cp(k):
        return (0. if k==0 else float(beta.ppf(.0125,k,n-k+1)),
                1. if k==n else float(beta.ppf(.9875,k+1,n-k)))
    wl,wh=cp(wins); ll,lh=cp(losses)
    return dict(wins=wins,losses=losses,delta_ler=delta,delta_ci_low=ll-wh,delta_ci_high=lh-wl,
                mcnemar_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
                interpretation='negative delta favors first decoder; conservative exact paired CI from Bonferroni Clopper-Pearson margins; p unadjusted')
