import contextlib,hashlib,importlib.metadata as md,io,json,warnings
from pathlib import Path
import numpy as np
from forecastscope.diagnostics import nonlinear as nl,seasonality as se
rows=[]
with warnings.catch_warnings(),contextlib.redirect_stdout(io.StringIO()):
 warnings.simplefilter('ignore')
 for level in (0.,.1,1e-14,1e12):
  a=np.full(128,level);r={'level':level}
  for name,fn in [('K',lambda:nl.zero_one_test(a)),('chaos',lambda:vars(nl.chaos_measures(a))),('report',lambda:nl.nonlinear_report(a,n_surrogates=1)),('rqa',lambda:vars(nl.recurrence_quantification(a))),('embedding',lambda:{'shape':list(nl.time_delay_embed(a).shape),'all_original_level':bool(np.all(nl.time_delay_embed(a)==level))}),('surrogate',lambda:{'all_original_level':bool(np.all(nl.iaaft_surrogate(a)==level))})]:
   try:
    with np.errstate(all='warn'):r[name]={'result':fn()}
   except Exception as e:r[name]={'error':type(e).__name__,'message':str(e)[:300]}
  rows.append(r)
 genuine=[]
 x=np.sin(2*np.pi*np.arange(128)/17)+np.random.default_rng(352).normal(0,.15,128)
 for u in (1e-9,1.,1e9):
  with np.errstate(all='warn'):r=nl.nonlinear_report(u*x)
  genuine.append({'unit':u,'report':r,'threshold_normalized':nl.recurrence_quantification(u*x).threshold/u})
 seasonal=[]
 x=np.sin(2*np.pi*np.arange(144)/12)+np.random.default_rng(81).normal(0,.2,144)+.01*np.arange(144)
 for u in (1e-9,1.,1e9):
  r=se.seasonality_report(u*x,candidate_periods=[12,24]);a=se.stl_decompose(u*x,12);b=se.mstl_decompose(u*x,[12,24]);seasonal.append({'unit':u,'strength':se.seasonal_strength(u*x,12),'spectral_period':r['periodogram']['dominant_period'],'spectral_power':r['periodogram']['dominant_power']/u**2,'welch_period':r['welch']['dominant_period'],'welch_power':r['welch']['dominant_power']/u**2,'stl':[a.strength_trend,a.strength_seasonal],'mstl':[b.strength_trend,b.strength_seasonal]})
 constants=[]
 for level in (0.,.1,1e-14,1e12):
  z=np.full(144,level);a=se.stl_decompose(z,12);b=se.mstl_decompose(z,[12,24]);constants.append({'level':level,'strength':se.seasonal_strength(z,12),'stl':[a.strength_trend,a.strength_seasonal],'mstl':[b.strength_trend,b.strength_seasonal]})
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [clean(x) for x in v]
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
print(json.dumps(clean({'state':'independent-observations','version':md.version('forecastscope'),'sources':{str(m.__file__):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (nl,se)},'nonlinear_constants':rows,'nonlinear_genuine':genuine,'seasonality_genuine':seasonal,'seasonality_constants':constants}),allow_nan=False))
