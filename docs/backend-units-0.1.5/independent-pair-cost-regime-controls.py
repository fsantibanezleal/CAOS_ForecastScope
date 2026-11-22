import contextlib,hashlib,importlib.metadata as md,io,json,warnings
from pathlib import Path
import numpy as np
from forecastscope.diagnostics import causality as ca,changepoints as cp,filters as fl,volatility as vo

def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,float) and not np.isfinite(v):return None
 if isinstance(v,np.generic):return v.item()
 return v
rng=np.random.default_rng(124);cause=rng.normal(size=320);effect=.9*np.roll(cause,2)+rng.normal(0,.7,320)
rng=np.random.default_rng(812);walk=np.cumsum(rng.normal(size=240));linked=2*walk+rng.normal(0,.2,240)
rng=np.random.default_rng(814);variance_change=rng.normal(size=240)*np.r_[np.ones(120),np.full(120,3.)]
rng=np.random.default_rng(328);regimes=np.repeat([0.,4.,0.,4.],60)+rng.normal(0,1.5,240)
t=np.arange(256);emd_signal=np.sin(2*np.pi*t/16)+.3*np.sin(2*np.pi*t/51)+.01*t
rng=np.random.default_rng(981);arch_signal=rng.normal(size=512)*np.tile(np.r_[np.ones(32),np.full(32,5.)],8)
pairs=[];univariate=[]
with warnings.catch_warnings(),contextlib.redirect_stdout(io.StringIO()):
 warnings.simplefilter('ignore')
 for xu,yu in ((1.,1.),(1e-9,1e9),(1e9,1e-9)):
  r=ca.granger_causality(xu*cause,yu*effect,max_lag=5);j=ca.cointegration(xu*walk,yu*linked)
  pairs.append({'units':[xu,yu],'granger':{'best_lag':r.best_lag,'pvalues':r.pvalues,'causes':r.causes},'cointegration':vars(j)})
 for u in (1e-9,1.,1e9):
  e=fl.emd(u*emd_signal,max_imfs=6);a=vo.arch_lm(u*arch_signal);p=cp.pelt(u*variance_change,model='normal',penalty=15);b=cp.binseg(u*variance_change,n_bkps=1,model='normal');r=cp.markov_regimes(u*regimes);c=cp.cusum_stability(u*variance_change)
  order=np.argsort(r.regime_means);probs=np.asarray(r.smoothed_probabilities)[:,order];tm=np.asarray(r.transition_matrix)[np.ix_(order,order)]
  univariate.append({'unit':u,'emd':{'imfs':e.imfs/u,'residual':e.residual/u,'reconstruction_error':float(np.max(np.abs((e.imfs.sum(axis=0)+e.residual)/u-emd_signal)))},'arch':vars(a),'pelt':p.breakpoints,'binseg':b.breakpoints,'markov':{'means':np.asarray(r.regime_means)[order]/u,'probabilities':probs,'transition':tm},'cusum':vars(c)})
record={'state':'independent-observations','version':md.version('forecastscope'),'sources':{str(m.__file__):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (ca,cp,fl,vo)},'fixture_f64_sha256':{k:hashlib.sha256(np.asarray(v,dtype='<f8').tobytes()).hexdigest() for k,v in {'cause':cause,'effect':effect,'walk':walk,'linked':linked,'variance_change':variance_change,'regimes':regimes,'emd':emd_signal,'arch':arch_signal}.items()},'pairs':pairs,'univariate':univariate}
print(json.dumps(clean(record),allow_nan=False))
