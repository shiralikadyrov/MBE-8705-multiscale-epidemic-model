"""Revised MBE-8705 experiments. Run from the package root: python code/revised_analysis.py.
No uploads or external data. All outputs are deterministic conditional on the recorded environment.
Positive within-host states are integrated in logarithmic coordinates, without artificial extinction.
Population flows and cumulative integrals are integrated with the same segmented ODE solver.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json, sys, time, platform
import numpy as np
import scipy, pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, least_squares
from scipy.stats import rankdata, t as student_t, qmc, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; FIG=ROOT/'figures'
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
@dataclass
class Params:
 eta:float=.05
 lam:float=10.
 dT:float=.01
 beta_h:float=2e-4
 dIh:float=.5
 kappa:float=.002
 p:float=50.
 c:float=5.
 r:float=.0001
 dX:float=.1
 eps_max:float=.95
 alpha_res:float=.0252
 beta0:float=1.25
 K:float=80.
 mu:float=1/(70*365)
 sigma:float=.2
 gamma:float=1/7
 N:float=1.
 u_max:float=.012
 w_inc:float=800.
 w_Ip:float=150.
 w_Vh:float=.35
 w_u:float=900.
 w_a:float=25.
 w_terminal:float=900.
 w_res:float=2.
 T_final:float=180.

def equilibrium(p,eps):
 q=p.p*(1-eps); Rh=p.beta_h*p.lam*q/(p.dT*p.dIh*p.c)
 if Rh<=1: return np.array([p.lam/p.dT,0.,0.,0.])
 I=p.lam/p.dIh-p.c*p.dT/(p.beta_h*q)
 if p.r*I<=p.dX: return np.array([p.c*p.dIh/(p.beta_h*q),I,q*I/p.c,0.])
 I=p.dX/p.r; V=q*I/p.c; T=p.lam/(p.dT+p.beta_h*V)
 return np.array([T,I,V,(p.beta_h*T*V/I-p.dIh)/p.kappa])

def rc(p,u,a,t):
 V=equilibrium(p,p.eps_max*a*np.exp(-p.alpha_res*t))[2]
 b=p.beta0*V/(V+p.K)
 return b*p.sigma/((p.sigma+p.mu)*(p.gamma+p.mu))*p.mu/(p.mu+u)

def heuristic(p,M=12):
 grid=np.linspace(0,p.T_final,M+1); us=np.linspace(0,p.u_max,9); aa=np.linspace(0,1,11)
 u=[]; a=[]
 for tj in grid[:-1]:
  candidates=[(rc(p,x,y,tj),p.w_u*x*x+p.w_a*y*y,float(x),float(y)) for x in us for y in aa]
  ok=[v for v in candidates if v[0]<=.95]
  best=min(ok,key=lambda v:(v[1],v[2],v[3])) if ok else min(candidates,key=lambda v:(v[0]+1e-4*v[1],v[2],v[3]))
  u.append(best[2]); a.append(best[3])
 return np.array(u),np.array(a)

def simulate(p,u,a,dense=False,rtol=1e-8,atol=1e-10,method='LSODA'):
 M=len(u); grid=np.linspace(0,p.T_final,M+1)
 state=np.r_[np.log([p.lam/p.dT,.05,5.,1.]),[.999,.0005,.0005,0.],0.,0.,0.]
 ts=[]; states=[]
 for j in range(M):
  uj=float(u[j]); aj=float(a[j])
  def rhs(t,y):
   T,I,V,X=np.exp(y[:4]); S,E,Ip,R=y[4:8]
   eps=p.eps_max*aj*np.exp(-p.alpha_res*t)
   b=p.beta0*V/(V+p.K); inc=b*S*Ip/p.N
   lh=np.array([p.lam/T-p.dT-p.beta_h*V,p.beta_h*np.exp(y[0]+y[2]-y[1])-p.dIh-p.kappa*X,p.p*(1-eps)*np.exp(y[1]-y[2])-p.c,p.r*I-p.dX])/p.eta
   cost=p.w_inc*inc+p.w_Ip*Ip+p.w_Vh*V+.5*p.w_u*uj**2+(.5*p.w_a+p.w_res*p.alpha_res)*aj**2
   return np.r_[lh,p.mu*p.N-inc-(p.mu+uj)*S,inc-(p.sigma+p.mu)*E,p.sigma*E-(p.gamma+p.mu)*Ip,p.gamma*Ip-p.mu*R+uj*S,inc,cost,V]
  sol=solve_ivp(rhs,(grid[j],grid[j+1]),state,method=method,rtol=rtol,atol=atol,dense_output=dense)
  if not sol.success or not np.isfinite(sol.y).all(): raise RuntimeError(sol.message)
  if dense:
   tt=np.linspace(grid[j],grid[j+1],int(round(grid[j+1]-grid[j])*10)+1)
   if j: tt=tt[1:]
   ts.extend(tt); states.extend(sol.sol(tt).T)
  state=sol.y[:,-1]
 result={'objective':float(state[9]+p.w_terminal*state[6]),'cum_incidence':float(state[8]),'mean_Vh':float(state[10]/p.T_final),'final_Ip':float(state[6]),'u':np.array(u),'a':np.array(a)}
 if dense:
  Y=np.array(states).T; t=np.array(ts); Y[:4]=np.exp(Y[:4]); idx=np.minimum(np.searchsorted(grid,t,side='right')-1,M-1)
  beta=p.beta0*Y[2]/(Y[2]+p.K); result.update(t=t,Y=Y,incidence=beta*Y[4]*Y[6],u_t=np.array(u)[idx],a_t=np.array(a)[idx],peak_Ip=float(Y[6].max()),mass_error=float(abs(Y[4:8].sum(axis=0)-1).max()))
 if dense and (np.min(result['Y'][4:8]) < -1e-8 or result['mass_error']>1e-7): raise RuntimeError('Population invariant check failed')
 return result

def optimize(p,M=12,seed=103,maxiter=60,n_starts=4,warm=None):
 uh,ah=heuristic(p,M); rng=np.random.default_rng(seed)
 starts=[np.r_[uh/p.u_max,ah],np.r_[np.ones(M),np.ones(M)],np.r_[np.zeros(M),np.ones(M)]]
 if warm is not None: starts[2]=warm
 starts.append(rng.uniform(0,1,2*M))
 best=None; logs=[]
 def fun(x): return simulate(p,x[:M]*p.u_max,x[M:])['objective']/10000
 for k,x0 in enumerate(starts[:n_starts]):
  start=time.time()
  res=minimize(fun,x0,method='L-BFGS-B',bounds=[(0,1)]*(2*M),options={'maxiter':maxiter,'maxfun':3500,'ftol':1e-10,'gtol':1e-5,'eps':1e-5,'maxls':30})
  val=fun(res.x)*10000
  logs.append({'start':k,'objective':val,'success':bool(res.success),'message':str(res.message),'nit':int(res.nit),'nfev':int(res.nfev),'seconds':time.time()-start})
  print('OPT',p.alpha_res,k,logs[-1],flush=True)
  if best is None or val<best[0]: best=(val,res.x.copy())
 # Retain an evaluated feasible initial policy if solver result is worse.
 for x in starts[:n_starts]:
  val=fun(x)*10000
  if val<best[0]: best=(val,x.copy())
 return best[1][:M]*p.u_max,best[1][M:],logs

def weekly(sim):
 edges=np.r_[np.arange(0,180,7),180.]
 cumulative=np.interp(edges,sim['t'],sim['Y'][8]); return (edges[:-1]+edges[1:])/2,np.diff(cumulative)

def write_trajectory(name,s):
 pd.DataFrame({'t':s['t'],**{x:s['Y'][i] for i,x in enumerate(['Th','Ih','Vh','Xh','S','Ep','Ip','R','cum_incidence','running_objective','integral_Vh'])},'u':s['u_t'],'a':s['a_t'],'incidence':s['incidence']}).to_csv(DATA/(name+'.csv'),index=False)

def main():
 p=Params()
 if '--figures-only' in sys.argv:
  figures(p,load_policies()); return
 if '--verify-only' in sys.argv:
  robustness(p,load_policies()); figures(p,load_policies()); return
 p=Params(); metadata={'parameters':asdict(p),'M':12,'optimizer':{'starts':4,'maxiter':40,'maxfun':3500,'seeds':list(range(100,106)),'eps_scaled':1e-5,'ftol':1e-10,'gtol':1e-5},'integration':{'method':'LSODA','rtol':1e-8,'atol':1e-10,'segmented':True,'positive_log_host_states':True},'alpha_grid':list(np.linspace(.003,.04,6)),'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__}
 metadata.update({'calibration': {'phi': 0.1, 'r_nb': 10, 'N_obs': 200000, 'data_seed': 42, 'bootstrap_seed': 11, 'bootstrap_requested': 40, 'bounds': [[0.001, 0.08], [0.8, 2], [20, 220]], 'start_vectors': [[0.004, 1, 50], [0.025, 1.5, 120], [0.05, 1.1, 180], [0.015, 1.7, 100]], 'truth': [0.01, 1.25, 80], 'controls': {'u': 0.0015, 'a': 0.5}}, 'sensitivity': {'latin_hypercube_n': 250, 'latin_hypercube_seed': 1, 'sobol_base_n': 256, 'sobol_seed': 2, 'sobol_bootstrap_n': 1000, 'sobol_bootstrap_seed': 3}, 'uncertainty': {'draws_per_alpha': 12, 'seed': 123, 'bootstrap_n': 2000, 'bootstrap_seeds': [1000, 1001, 1002, 1003, 1004, 1005]}})
 (DATA/'run_manifest.json').write_text(json.dumps(metadata,indent=2))
 uh,ah=heuristic(p); z=np.zeros(12); one=np.ones(12)
 if '--pilot' in sys.argv:
  for name,u,a in [('no',z,z),('heuristic',uh,ah),('max',p.u_max*one,one),('early',z,np.r_[1.,np.zeros(11)])]:
   t=time.time(); s=simulate(p,u,a,True); print(name,{k:s[k] for k in ['objective','cum_incidence','peak_Ip','mass_error']},'seconds',time.time()-t,flush=True)
  print('Rc0',rc(p,0,0,0),'Rch',rc(p,uh[0],ah[0],0),'controls',uh,ah,flush=True); return
 rows=[]; policies={}; warm=None; all_logs={}
 from concurrent.futures import ProcessPoolExecutor
 with ProcessPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(optimize,replace(p,alpha_res=float(alpha)),12,100+i,40,4) for i,alpha in enumerate(np.linspace(.003,.04,6))]
  runs=[f.result() for f in futures]
 for i,alpha in enumerate(np.linspace(.003,.04,6)):
  pp=replace(p,alpha_res=float(alpha)); uo,ao,logs=runs[i]
  warm=np.r_[uo/p.u_max,ao]; all_logs[str(alpha)]=logs; uh,ah=heuristic(pp)
  for name,u,a in [('no_control',z,z),('heuristic',uh,ah),('maximal',p.u_max*one,one),('numerically_optimized',uo,ao)]:
   s=simulate(pp,u,a,True); policies[(float(alpha),name)]=s
   rows.append({'alpha_res':alpha,'policy':name,**{k:s[k] for k in ['objective','cum_incidence','peak_Ip','mean_Vh','final_Ip','mass_error']}})
   write_trajectory(f'alpha_{i}_{name}',s)
  pd.DataFrame(rows).to_csv(DATA/'policy_results.csv',index=False)
  (DATA/'optimizer_logs.json').write_text(json.dumps(all_logs,indent=2))
  np.savez(DATA/'policy_cache.npz',**{f'{i}_{name}_{k}':policies[(float(alpha),name)][k] for i,alpha in enumerate(np.linspace(.003,.04,6)) for name in ['heuristic','numerically_optimized'] if (float(alpha),name) in policies for k in ['u','a']})
 robustness(p,policies)
 if '--policies-only' not in sys.argv: additional(p,policies)
 figures(p,policies)

# Additional analyses and plotting are defined below.

def additional(p,policies=None):
 from scipy.stats import nbinom
 bounds=np.array([[.001,.05],[.9,1.6],[40,140],[1/14,1/4],[1/8,1/3],[.70,.99]])
 names=['alpha_res','beta0','K','gamma','sigma','eps_max']
 def response(x,mode):
  pp=replace(p,**dict(zip(names,map(float,x))))
  u,a=heuristic(pp) if mode=='heuristic' else (np.full(12,.0015),np.full(12,.8))
  return simulate(pp,u,a)['cum_incidence']
 sample=qmc.LatinHypercube(6,seed=1).random(250); X=qmc.scale(sample,bounds[:,0],bounds[:,1])
 records=[]
 for mode in ['heuristic','fixed_treatment']:
  Y=np.array([response(x,mode) for x in X]); pd.DataFrame(np.c_[X,Y],columns=names+['cum_incidence']).to_csv(DATA/('sensitivity_samples_'+mode+'.csv'),index=False)
  Xr=np.apply_along_axis(rankdata,0,X); Yr=rankdata(Y)
  for j,name in enumerate(names):
   Z=np.c_[np.ones(250),np.delete(Xr,j,axis=1)]; rx=Xr[:,j]-Z@np.linalg.lstsq(Z,Xr[:,j],rcond=None)[0]; ry=Yr-Z@np.linalg.lstsq(Z,Yr,rcond=None)[0]
   corr=float(np.corrcoef(rx,ry)[0,1]); dof=250-6-1
   pv=float(2*student_t.sf(abs(corr)*np.sqrt(dof/max(1-corr*corr,1e-15)),dof))
   records.append({'mode':mode,'parameter':name,'PRCC':corr,'p_value':pv})
  print('PRCC',mode,records[-6:],flush=True)
 pd.DataFrame(records).to_csv(DATA/'prcc_results.csv',index=False)
 # Scrambled Sobol A/B design, N=256; explicit estimators and bootstrap half-widths.
 ab=qmc.Sobol(12,scramble=True,seed=2).random_base2(8); A=qmc.scale(ab[:,:6],bounds[:,0],bounds[:,1]); B=qmc.scale(ab[:,6:],bounds[:,0],bounds[:,1])
 ya=np.array([response(x,'heuristic') for x in A]); yb=np.array([response(x,'heuristic') for x in B]); yab=[]
 for j in range(6):
  c=A.copy(); c[:,j]=B[:,j]; yab.append(np.array([response(x,'heuristic') for x in c])); print('SOBOL',j,flush=True)
 yab=np.array(yab); np.savez(DATA/'sobol_samples.npz',A=A,B=B,YA=ya,YB=yb,YAB=yab)
 def indices(ix):
  a=ya[ix]; b=yb[ix]; v=np.var(np.r_[a,b],ddof=1); mean=np.mean(np.r_[a,b]); c=yab[:,ix]
  return np.mean((b-mean)*(c-a),axis=1)/v,.5*np.mean((a-c)**2,axis=1)/v
 s1,st=indices(np.arange(256)); rng=np.random.default_rng(3); reps=[indices(rng.integers(0,256,256)) for _ in range(1000)]; boot=np.array(reps)
 pd.DataFrame({'parameter':names,'S1':s1,'ST':st,'S1_halfwidth':1.96*boot[:,0,:].std(axis=0,ddof=1),'ST_halfwidth':1.96*boot[:,1,:].std(axis=0,ddof=1)}).to_csv(DATA/'sobol_results.csv',index=False)
 calibration(p)
 print('ADDITIONAL DONE',flush=True)


def robustness(p,policies):
 rng=np.random.default_rng(123); rows=[]; summary=[]
 for alpha in np.linspace(.003,.04,6):
  pp=replace(p,alpha_res=float(alpha)); h=policies[(float(alpha),'heuristic')]; o=policies[(float(alpha),'numerically_optimized')]
  for i in range(12):
   pert=replace(pp,beta0=pp.beta0*rng.uniform(.9,1.1),K=pp.K*rng.uniform(.85,1.15),gamma=pp.gamma*rng.uniform(.9,1.1),sigma=pp.sigma*rng.uniform(.9,1.1),eps_max=float(np.clip(pp.eps_max*rng.uniform(.95,1.05),.65,.99)),alpha_res=float(np.clip(pp.alpha_res*rng.uniform(.9,1.1),.001,.1)))
   jh=simulate(pert,h['u'],h['a'])['objective']; jo=simulate(pert,o['u'],o['a'])['objective']; rows.append({'alpha_res':alpha,'mc_id':i,'J_heuristic':jh,'J_optimized':jo,'relative_improvement':100*(jh-jo)/jh,**{'draw_'+k:v for k,v in asdict(pert).items()}})
  d=pd.DataFrame(rows).query('alpha_res == @alpha'); rel=d.relative_improvement.to_numpy(); rr=np.random.default_rng(1000+len(summary)); boots=np.mean(rr.choice(rel,(2000,12),replace=True),axis=1); ci=np.percentile(boots,[2.5,97.5]); diff=d.J_heuristic-d.J_optimized
  pv=1. if np.all(abs(diff)<1e-12) else float(wilcoxon(diff,method='exact').pvalue)
  summary.append({'alpha_res':alpha,'mean_relative_improvement':float(rel.mean()),'lower':float(ci[0]),'upper':float(ci[1]),'wilcoxon_p':pv,'n':12})
 pd.DataFrame(rows).to_csv(DATA/'uncertainty_draws.csv',index=False); pd.DataFrame(summary).to_csv(DATA/'uncertainty_summary.csv',index=False)
 heat=[]
 for alpha in np.linspace(.003,.04,6):
  h=policies[(float(alpha),'heuristic')]; o=policies[(float(alpha),'numerically_optimized')]
  for q in [.05,.1,.2]:
   rr=np.random.default_rng(int(round(alpha*1e5))+int(100*q)); vals=[]
   for i in range(6):
    base=replace(p,alpha_res=float(alpha)); pert=replace(base,beta0=base.beta0*rr.uniform(.9,1.1),K=base.K*rr.uniform(.85,1.15),gamma=base.gamma*rr.uniform(.9,1.1),sigma=base.sigma*rr.uniform(.9,1.1),eps_max=float(np.clip(base.eps_max*rr.uniform(.95,1.05),.65,.99)),alpha_res=float(np.clip(base.alpha_res*rr.uniform(.9,1.1),.001,.1)))
    pert=replace(pert,beta0=pert.beta0*(1+rr.normal(0,.05+q)),alpha_res=float(np.clip(pert.alpha_res*(1+rr.normal(0,q)),.001,.1)))
    if pert.beta0<=0: raise RuntimeError('Invalid transmission perturbation')
    jh=simulate(pert,h['u'],h['a'])['objective']; jo=simulate(pert,o['u'],o['a'])['objective']; val=100*(jh-jo)/jh; vals.append(val)
    heat.append({'alpha_res':alpha,'perturbation_q':q,'replicate':i,'relative_improvement':val})
 pd.DataFrame(heat).to_csv(DATA/'heatmap_draws.csv',index=False)
 # Independent solver/tolerance validation of fixed computed controls.
 rows=[]
 for alpha in np.linspace(.003,.04,6):
  pp=replace(p,alpha_res=float(alpha)); s=policies[(float(alpha),'numerically_optimized')]
  for method,tol in [('LSODA',1e-10),('Radau',1e-9)]:
   chk=simulate(pp,s['u'],s['a'],True,rtol=tol,atol=tol*.01,method=method)
   rows.append({'alpha_res':alpha,'method':method,'rtol':tol,'objective':chk['objective'],'relative_objective_difference':abs(chk['objective']-s['objective'])/max(1,abs(s['objective'])),'cum_incidence':chk['cum_incidence'],'max_Ip_difference':float(abs(chk['Y'][6]-s['Y'][6]).max()),'mass_error':chk['mass_error']})
 pd.DataFrame(rows).to_csv(DATA/'solver_checks.csv',index=False)


def figures(p,policies):
 cols={'no_control':'#777777','heuristic':'#234f77','maximal':'#319477','numerically_optimized':'#d34b2c'}
 labels={'no_control':'No control','heuristic':'Threshold heuristic','maximal':'Constant maximal','numerically_optimized':'Numerically optimized'}
 alpha=float(np.linspace(.003,.04,6)[3]); fig,axs=plt.subplots(2,2,figsize=(8,6),layout='constrained')
 for name in cols:
  s=policies[(alpha,name)]; axs[0,0].plot(s['t'],s['Y'][6],label=labels[name],color=cols[name]); axs[0,1].plot(s['t'],s['Y'][2],color=cols[name]); mid,inc=weekly(s); axs[1,1].plot(mid,inc,color=cols[name])
 for name in ['heuristic','numerically_optimized']:
  s=policies[(alpha,name)]; axs[1,0].step(s['t'],s['a_t'],color=cols[name],label=labels[name]+': treatment'); axs[1,0].step(s['t'],s['u_t']/p.u_max,color=cols[name],ls='--',label=labels[name]+': vaccination / maximum')
 axs[0,0].legend(fontsize=7); axs[1,0].legend(fontsize=7); axs[0,1].set_yscale('symlog',linthresh=.01)
 for ax,title,y in zip(axs.flat,['A. Infectious prevalence','B. Representative viral load','C. Applied controls','D. Weekly new exposures'],['$I_p$','$V_h$','Intensity / capacity','Fraction per interval']): ax.set(title=title,xlabel='Day',ylabel=y); ax.grid(alpha=.2)
 fig.savefig(FIG/'fig1_policy_comparison.pdf'); plt.close(fig)
 df=pd.read_csv(DATA/'policy_results.csv'); fig,axs=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
 for name in cols:
  d=df[df.policy==name]; axs[0].plot(d.alpha_res,d.cum_incidence,'o-',color=cols[name],label=labels[name]); axs[1].plot(d.alpha_res,d.objective,'o-',color=cols[name])
 axs[0].legend(fontsize=7); axs[0].set(ylabel='Cumulative new exposures',title='A. Epidemic burden'); axs[1].set(ylabel='Objective $J$',title='B. Composite cost')
 for ax in axs: ax.set_xlabel('Resistance erosion rate (per day)'); ax.grid(alpha=.2)
 fig.savefig(FIG/'fig2_alpha_sweep.pdf'); plt.close(fig)
 if (DATA/'heatmap_draws.csv').exists():
  d=pd.read_csv(DATA/'heatmap_draws.csv').groupby(['perturbation_q','alpha_res']).relative_improvement.mean().unstack(); fig,ax=plt.subplots(figsize=(6.8,2.8),layout='constrained'); im=ax.imshow(d.values,aspect='auto',cmap='RdBu_r',vmin=-100,vmax=100); ax.set_xticks(range(6),[f'{x:.4f}' for x in d.columns]); ax.set_yticks(range(3),[f'{x:.2f}' for x in d.index]); ax.set(xlabel='Resistance erosion rate (per day)',ylabel='Parameter perturbation q')
  for i in range(3):
   for j in range(6): ax.text(j,i,f'{d.values[i,j]:.1f}',ha='center',va='center',color='black')
  fig.colorbar(im,ax=ax,label='Mean objective improvement (%)'); fig.savefig(FIG/'fig3_heatmap.pdf'); plt.close(fig)
 if (DATA/'calibration_curves.csv').exists():
  d=pd.read_csv(DATA/'calibration_curves.csv'); fig,ax=plt.subplots(figsize=(6.8,3.5),layout='constrained'); ax.fill_between(d.week_mid,d.lower,d.upper,color='#d34b2c',alpha=.2,label='95% fitted-mean bootstrap interval'); ax.plot(d.week_mid,d.true_fraction,'--',color='#319477',label='Generating mean'); ax.plot(d.week_mid,d.fit_fraction,color='#d34b2c',label='Fitted mean'); ax.scatter(d.week_mid,d.observed_fraction,s=15,color='black',label='NB observations'); ax.set(xlabel='Day (weekly midpoint)',ylabel='Weekly new exposures / population'); ax.legend(fontsize=8); fig.savefig(FIG/'fig4_calibration.pdf'); plt.close(fig)
 if (DATA/'prcc_results.csv').exists():
  d=pd.read_csv(DATA/'prcc_results.csv'); fig,ax=plt.subplots(figsize=(6.8,3.5),layout='constrained'); names=d.parameter.unique(); x=np.arange(6)
  for i,mode in enumerate(['heuristic','fixed_treatment']):
   s=d[d['mode']==mode].set_index('parameter').loc[names]; ax.bar(x+(i-.5)*.35,s.PRCC,width=.35,label='Threshold heuristic' if i==0 else 'Fixed nonzero treatment',color=['#234f77','#d34b2c'][i])
  ax.set_xticks(x,['alpha','beta0','K','gamma','sigma','eps_max']); ax.set(ylim=(-1,1),ylabel='PRCC'); ax.axhline(0,color='black',lw=.5); ax.legend(fontsize=8); fig.savefig(FIG/'fig5_prcc.pdf'); plt.close(fig)
 if (DATA/'uncertainty_summary.csv').exists():
  d=pd.read_csv(DATA/'uncertainty_summary.csv'); fig,ax=plt.subplots(figsize=(6.8,3.5),layout='constrained'); ax.errorbar(d.alpha_res,d.mean_relative_improvement,yerr=np.vstack([d.mean_relative_improvement-d.lower,d.upper-d.mean_relative_improvement]),fmt='o-',color='#d34b2c',capsize=3); ax.axhline(0,color='black',ls='--',lw=.7); ax.set(xlabel='Resistance erosion rate (per day)',ylabel='Objective improvement (%)'); fig.savefig(FIG/'fig6_uncertainty.pdf'); plt.close(fig)

def fit_calibration(counts,start,maxiter=80):
 from scipy.stats import nbinom
 true=replace(Params(),alpha_res=.01); u=np.full(12,.0015); a=np.full(12,.5)
 norm=np.array([.01,1.25,80.]); lb=np.array([.001,.8,20]); ub=np.array([.08,2.,220]); shape=10.; scale=200000
 def loss(x):
  th=x*norm; ss=simulate(replace(true,alpha_res=th[0],beta0=th[1],K=th[2]),u,a,True)
  mu=np.maximum(scale*weekly(ss)[1],1e-12)
  return -float(nbinom.logpmf(counts,shape,shape/(shape+mu)).sum())/len(counts)
 res=minimize(loss,np.array(start)/norm,method='L-BFGS-B',bounds=list(zip(lb/norm,ub/norm)),options={'maxiter':maxiter,'maxfun':1200,'eps':1e-4,'ftol':1e-10,'gtol':1e-5,'maxls':30})
 return {'start':list(map(float,start)),'theta':(res.x*norm).tolist(),'negative_log_likelihood':float(res.fun*len(counts)),'success':bool(res.success),'message':str(res.message),'nfev':int(res.nfev)}

def bootstrap_calibration_task(task):
 i,counts,hat=task; log=fit_calibration(counts,hat,maxiter=60); log['replicate']=i
 if log['success']:
  th=log['theta']; pp=replace(Params(),alpha_res=th[0],beta0=th[1],K=th[2]); curve=weekly(simulate(pp,np.full(12,.0015),np.full(12,.5),True))[1].tolist()
 else: curve=None
 return log,curve

def calibration(p,resume=False):
 from scipy.stats import nbinom
 from concurrent.futures import ProcessPoolExecutor
 true=replace(p,alpha_res=.01); u=np.full(12,.0015); a=np.full(12,.5); sim=simulate(true,u,a,True); mids,means=weekly(sim); scale=200000; phi=.10; shape=1/phi
 rng=np.random.default_rng(42); obs=rng.negative_binomial(shape,shape/(shape+scale*means))
 pd.DataFrame({'week_mid':mids,'mean_count':scale*means,'observed_count':obs,'mean_fraction':means}).to_csv(DATA/'synthetic_observations.csv',index=False)
 starts=[[.004,1.,50.],[.025,1.5,120.],[.05,1.1,180.],[.015,1.7,100.]]
 if resume and (DATA/'calibration_initial_fits.json').exists(): fitlogs=json.loads((DATA/'calibration_initial_fits.json').read_text())
 else:
  fitlogs=[]
  for start in starts:
   log=fit_calibration(obs,start); fitlogs.append(log); print('FIT',log,flush=True)
  (DATA/'calibration_initial_fits.json').write_text(json.dumps(fitlogs,indent=2))
 best=min(fitlogs,key=lambda x:x['negative_log_likelihood']); hat=np.array(best['theta']); hatsim=simulate(replace(true,alpha_res=hat[0],beta0=hat[1],K=hat[2]),u,a,True); fitted=weekly(hatsim)[1]
 rng=np.random.default_rng(11); tasks=[(i,rng.negative_binomial(shape,shape/(shape+scale*fitted)),hat) for i in range(40)]
 with ProcessPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(bootstrap_calibration_task,x) for x in tasks]; out=[]
  for i,f in enumerate(futures):
   item=f.result(); out.append(item); print('BOOT',i,item[0]['success'],flush=True)
   (DATA/'bootstrap_progress.json').write_text(json.dumps([x[0] for x in out],indent=2))
 curves=np.array([curve for log,curve in out if curve is not None]); theta_boot=np.array([log['theta'] for log,curve in out if curve is not None]); bootlogs=[x[0] for x in out]
 if len(curves)<2: raise RuntimeError('Insufficient successful bootstrap fits')
 lo,hi=np.percentile(curves,[2.5,97.5],axis=0)
 pd.DataFrame({'week_mid':mids,'true_fraction':means,'observed_fraction':obs/scale,'fit_fraction':fitted,'lower':lo,'upper':hi}).to_csv(DATA/'calibration_curves.csv',index=False)
 pd.DataFrame(theta_boot,columns=['alpha_res','beta0','K']).to_csv(DATA/'calibration_bootstrap_parameters.csv',index=False)
 (DATA/'calibration_logs.json').write_text(json.dumps({'truth':[.01,1.25,80.],'best_theta':hat.tolist(),'best_success':best['success'],'fits':fitlogs,'bootstrap_requested':40,'bootstrap_successful':len(curves),'bootstrap_logs':bootlogs,'phi':phi,'r_nb':shape,'population_scale':scale},indent=2))
 deriv=[]; norm=np.array([.01,1.25,80.])
 for j in range(3):
  plus=hat.copy(); minus=hat.copy(); step=1e-3*norm[j]; plus[j]+=step; minus[j]-=step
  def curve(th): return weekly(simulate(replace(true,alpha_res=th[0],beta0=th[1],K=th[2]),u,a,True))[1]*scale
  deriv.append((curve(plus)-curve(minus))/(2*step)*norm[j])
 Jac=np.array(deriv).T/np.sqrt(scale*fitted+(scale*fitted)**2*phi)[:,None]
 sv=np.linalg.svd(Jac,compute_uv=False); (DATA/'calibration_conditioning.json').write_text(json.dumps({'scaled_weighted_singular_values':sv.tolist(),'condition_number':float(sv[0]/sv[-1]),'note':'Weighted local mean-response Jacobian at the lowest returned likelihood candidate, parameters scaled by nominal values; not a structural identifiability test.'},indent=2))
 print('CALIBRATION DONE',flush=True)

def load_policies():
 results=pd.read_csv(DATA/'policy_results.csv'); policies={}; cache=np.load(DATA/'policy_cache.npz')
 for i,alpha in enumerate(np.linspace(.003,.04,6)):
  for name in ['no_control','heuristic','maximal','numerically_optimized']:
   d=pd.read_csv(DATA/f'alpha_{i}_{name}.csv'); row=results[(abs(results.alpha_res-alpha)<1e-12)&(results.policy==name)].iloc[0]
   if name in ['heuristic','numerically_optimized']: u=cache[f'{i}_{name}_u']; a=cache[f'{i}_{name}_a']
   elif name=='no_control': u=np.zeros(12); a=np.zeros(12)
   else: u=np.full(12,Params().u_max); a=np.ones(12)
   policies[(float(alpha),name)]={'t':d.t.to_numpy(),'Y':d[['Th','Ih','Vh','Xh','S','Ep','Ip','R','cum_incidence','running_objective','integral_Vh']].to_numpy().T,'u':u,'a':a,'u_t':d.u.to_numpy(),'a_t':d.a.to_numpy(),'incidence':d.incidence.to_numpy(),**{k:float(row[k]) for k in ['objective','cum_incidence','peak_Ip','mean_Vh','final_Ip','mass_error']}}
 return policies

if __name__=='__main__':
 main()
