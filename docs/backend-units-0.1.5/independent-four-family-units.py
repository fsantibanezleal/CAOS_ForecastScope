import contextlib, hashlib, importlib.metadata as md, io, json, warnings
from pathlib import Path
import numpy as np
from forecastscope.diagnostics import autocorrelation as ac, distribution as d, fractal as f, sequence as sq

rng=np.random.default_rng(1903)
x=2+np.sin(2*np.pi*np.arange(256)/19)+rng.normal(0,.3,256)
y=np.roll(x,3)+np.random.default_rng(111).normal(0,.2,256)
rows=[]
with warnings.catch_warnings(),contextlib.redirect_stdout(io.StringIO()):
 warnings.simplefilter('ignore')
 for u,yu in [(1e-9,1e9),(1.,1.),(1e9,1e-9)]:
  with np.errstate(all='warn'):
   a=ac.autocorrelation_report(u*x);s=d.summary(u*x);c=d.complexity(u*x);k=d.catch22_features(u*x)['features'];h=f.hurst(u*x);mf=f.mfdfa(u*x);fd=f.fractal_dimension(u*x);rho=f.dcca_coefficient(u*x,yu*y);kd=d.kde_curve(u*x);qq=d.qq_points(u*x);mse=sq.multiscale_entropy(u*x)
   normalized={'acf':[r['value'] for r in a['acf']],'pacf':[r['value'] for r in a['pacf']], 'lb_stat':a['ljung_box']['stat'],'lb_p':a['ljung_box']['pvalue'],'bp_stat':a['box_pierce']['stat'],'bp_p':a['box_pierce']['pvalue'],'dw':a['durbin_watson'],'mean':s.mean/u,'std':s.std/u,'skew':s.skewness,'kurt':s.kurtosis_excess,'jb':s.jarque_bera_pvalue,'shapiro':s.shapiro_pvalue,'sampen':c.sample_entropy,'perm':c.permutation_entropy,'spectral':c.spectral_entropy,'bds':c.bds_pvalue,'catch22':{key:val/u if key in ('DN_Mean','DN_Spread_Std') else val for key,val in k.items()},'kde_grid':(np.asarray(kd['grid'])/u).tolist(),'kde_density':(np.asarray(kd['density'])*u).tolist(),'qq_sample':(np.asarray(qq['sample'])/u).tolist(),'qq_slope':qq['slope']/u,'qq_intercept':qq['intercept']/u,'qq_r':qq['r'],'hurst':h.hurst_rs,'dfa':h.dfa_alpha,'mfwidth':mf.width,'hq':mf.hq.tolist(),'higuchi':fd.higuchi,'katz':fd.katz,'petrosian':fd.petrosian,'dcca_independent_units':rho['rho_dcca'],'mse':mse['scales'],'time_reversal':[{**sq.time_reversal(u*x,lag),'raw':sq.time_reversal(u*x,lag)['raw']/u**3} for lag in (1,2,7)]}
   for r in normalized['mse']:r['tolerance']/=u
   rows.append({'unit':u,'y_unit':yu,'normalized':normalized})
 constants=[]
 for level in (0.,.1,1e-14,1e12):
  z=np.full(256,level);r={'level':level}
  for name,fn in [('acf',lambda:ac.acf(z).values[:4].tolist()),('ljung_box',lambda:(ac.ljung_box(z).stat,ac.ljung_box(z).pvalue)),('kde',lambda:d.kde_curve(z)['density'][:2]),('hurst',lambda:(f.hurst(z).hurst_rs,f.hurst(z).dfa_alpha)),('dcca',lambda:f.dcca_coefficient(z,z)['rho_dcca'])]:
   try:
    with np.errstate(all='warn'):r[name]={'result':fn()}
   except Exception as e:r[name]={'error':type(e).__name__,'message':str(e)}
  constants.append(r)
 old=np.geterr();np.seterr(all='warn');before=np.geterr()
 try:
  f.hurst(np.zeros(256));after=np.geterr()
 except Exception as e:after=np.geterr()
 finally:np.seterr(**old)
 profiles=[]
 z=np.sin(2*np.pi*np.arange(128)/17)+np.random.default_rng(4201).normal(0,.2,128)
 for u in (1e-9,1.,1e9):
  r=sq.matrix_profile(u*z,window=12);profiles.append({'unit':u,'result':r})
 const_profiles=[]
 for level in (.1,1e-14,1e12):
  r=sq.matrix_profile(np.full(128,level),window=12);const_profiles.append({'level':level,'result':r})

def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
record={'state':'independent-observations','version':md.version('forecastscope'),'sources':{str(m.__file__):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (ac,d,f,sq)},'nonconstant':rows,'constants':constants,'hurst_error_settings':{'before':before,'after':after},'matrix_profiles':profiles,'constant_matrix_profiles':const_profiles}
print(json.dumps(clean(record),allow_nan=False))
