"""Supplementary minor-revision calculations; run from archive root.
Uses the fixed original policies, the archived uncertainty draws and the original
heatmap seeds. Also reoptimizes at w_V=0 with the recorded four-start protocol.
"""
from dataclasses import replace,asdict
from pathlib import Path
from scipy.integrate import simpson
from scipy.stats import wilcoxon
from scipy.optimize import minimize
import numpy as np, pandas as pd, json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revised_analysis import Params,simulate,load_policies,heuristic
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; FIG=ROOT/'figures'; alphas=np.linspace(.003,.04,6); M=12

def costs(p,s):
 u,a=s['u'],s['a']; T=p.T_final; step=T/len(u)
 c={'new_exposures':p.w_inc*s['cum_incidence'],
    'prevalence':p.w_Ip*float(simpson(s['Y'][6],x=s['t'])),
    'viral_load':p.w_Vh*s['Y'][10,-1],
    'vaccination':step*.5*p.w_u*float(u@u),
    'treatment':step*.5*p.w_a*float(a@a),
    'resistance_penalty':step*p.w_res*p.alpha_res*float(a@a),
    'terminal':p.w_terminal*s['final_Ip']}
 discrepancy=sum(c.values())-s['objective']
 if abs(discrepancy)>0.01:raise ValueError(f'Objective breakdown discrepancy {discrepancy}')
 return c,discrepancy

def main():
 p=Params();pol=load_policies(); alpha=float(alphas[3]); pb=replace(p,alpha_res=alpha)
 baserows=[]
 for label in ['no_control','heuristic','maximal','numerically_optimized']:
  s=pol[(alpha,label)]; c,e=costs(pb,s)
  baserows.append({'policy':label,**c,'J_reconstructed':sum(c.values()),'J_solver':s['objective'],'abs_discrepancy':abs(e),'cum_incidence':s['cum_incidence']})
 bd=pd.DataFrame(baserows);bd.to_csv(DATA/'objective_decomposition_baseline.csv',index=False);print('BASE COMPONENTS',bd.to_string(index=False),flush=True)
 # Original schedules under changed objective, PLUS a new optimization under that objective.
 p0=replace(pb,w_Vh=0.)
 zero=[]
 for label in ['no_control','heuristic','maximal','numerically_optimized']:
  old=pol[(alpha,label)];s=simulate(p0,old['u'],old['a'],True)
  zero.append({'policy':label,'J':s['objective'],'cum_incidence':s['cum_incidence'],'peak_Ip':s['peak_Ip'],**costs(p0,s)[0]})
 t=time.time(); uo,ao,logs=optimize_viral_zero(p0,pol[(alpha,'numerically_optimized')]);s=simulate(p0,uo,ao,True)
 zero.append({'policy':'optimized_for_wV_zero','J':s['objective'],'cum_incidence':s['cum_incidence'],'peak_Ip':s['peak_Ip'],**costs(p0,s)[0]})
 zd=pd.DataFrame(zero);zd.to_csv(DATA/'policy_comparison_wV_zero.csv',index=False)
 np.savez(DATA/'policy_wV_zero.npz',u=uo,a=ao)
 (DATA/'optimizer_logs_wV_zero.json').write_text(json.dumps({'elapsed_seconds':time.time()-t,'starts':logs,'selected_J':s['objective']},indent=2))
 print('W_V ZERO',zd[['policy','J','cum_incidence','peak_Ip']].to_string(index=False),flush=True)
 # Same nominal resistance sweep, now with a genuinely active comparator.
 prior=pd.read_csv(DATA/'policy_results.csv'); rows=[]
 for al in alphas:
  d=prior[np.isclose(prior.alpha_res,al)].set_index('policy')
  o=d.loc['numerically_optimized']; h=d.loc['heuristic']; ma=d.loc['maximal']
  rows.append({'alpha_res':al,'J_optimized':o.objective,'J_heuristic':h.objective,'J_maximal':ma.objective,
               'improvement_vs_heuristic_pct':100*(h.objective-o.objective)/h.objective,
               'improvement_vs_maximal_pct':100*(ma.objective-o.objective)/ma.objective,
               'incidence_optimized':o.cum_incidence,'incidence_maximal':ma.cum_incidence})
 pd.DataFrame(rows).to_csv(DATA/'comparisons_vs_maximal_sweep.csv',index=False)
 # The same 72 archived perturbations: no redrawing or policy reoptimization.
 old=pd.read_csv(DATA/'uncertainty_draws.csv'); results=[];summary=[]
 for al in alphas:
  df=old[np.isclose(old.alpha_res,al)].sort_values('mc_id');pp=replace(p,alpha_res=float(al))
  for row in df.itertuples():
   params={f[5:]:getattr(row,f) for f in old.columns if f.startswith('draw_')}
   pert=replace(pp,**params)
   mx=pol[(float(al),'maximal')]; jo=row.J_optimized
   jm=simulate(pert,mx['u'],mx['a'])['objective']
   results.append({'alpha_res':al,'mc_id':row.mc_id,'J_maximal':jm,'J_optimized':jo,'relative_improvement_vs_maximal_pct':100*(jm-jo)/jm})
  d=pd.DataFrame(results).query('alpha_res == @al'); vals=d.relative_improvement_vs_maximal_pct.to_numpy();rng=np.random.default_rng(1000+len(summary))
  boots=np.mean(rng.choice(vals,(2000,12),replace=True),axis=1);lo,hi=np.percentile(boots,[2.5,97.5]);diff=d.J_maximal-d.J_optimized
  summary.append({'alpha_res':al,'mean_relative_improvement_vs_maximal_pct':vals.mean(),'lower':lo,'upper':hi,'wilcoxon_p':float(wilcoxon(diff,method='exact').pvalue),'positive_pairs':int((diff>0).sum()),'n':len(d)})
 pd.DataFrame(results).to_csv(DATA/'uncertainty_vs_maximal_draws.csv',index=False)
 sd=pd.DataFrame(summary);sd.to_csv(DATA/'uncertainty_vs_maximal_summary.csv',index=False)
 print('UNCERTAINTY MAX',sd.to_string(index=False),flush=True)
 # Repeat exact existing heatmap sampling stream. Assert old comparator values match archives.
 prev=pd.read_csv(DATA/'heatmap_draws.csv');heat=[]
 for al in alphas:
  h=pol[(float(al),'heuristic')];o=pol[(float(al),'numerically_optimized')];mx=pol[(float(al),'maximal')]
  for q in [.05,.1,.2]:
   rr=np.random.default_rng(int(round(al*1e5))+int(100*q))
   for i in range(6):
    pp=replace(p,alpha_res=float(al));pert=replace(pp,beta0=pp.beta0*rr.uniform(.9,1.1),K=pp.K*rr.uniform(.85,1.15),gamma=pp.gamma*rr.uniform(.9,1.1),sigma=pp.sigma*rr.uniform(.9,1.1),eps_max=float(np.clip(pp.eps_max*rr.uniform(.95,1.05),.65,.99)),alpha_res=float(np.clip(pp.alpha_res*rr.uniform(.9,1.1),.001,.1)))
    pert=replace(pert,beta0=pert.beta0*(1+rr.normal(0,.05+q)),alpha_res=float(np.clip(pert.alpha_res*(1+rr.normal(0,q)),.001,.1)))
    jo=simulate(pert,o['u'],o['a'])['objective'];jh=simulate(pert,h['u'],h['a'])['objective'];jm=simulate(pert,mx['u'],mx['a'])['objective'];v=100*(jh-jo)/jh
    target=prev[(np.isclose(prev.alpha_res,al)) & (np.isclose(prev.perturbation_q,q)) & (prev.replicate==i)].relative_improvement.iloc[0]
    if abs(v-target)>1e-7:raise ValueError(f'Heatmap sampling mismatch {al} {q} {i}: {v} != {target}')
    heat.append({'alpha_res':al,'perturbation_q':q,'replicate':i,'J_optimized':jo,'J_heuristic':jh,'J_maximal':jm,'relative_improvement_vs_heuristic_pct':v,'relative_improvement_vs_maximal_pct':100*(jm-jo)/jm,**{'draw_'+k:v for k,v in asdict(pert).items()}})
 hd=pd.DataFrame(heat);hd.to_csv(DATA/'heatmap_comparisons_two_baselines.csv',index=False)
 print('HEATMAP RANGES',hd.groupby(['perturbation_q','alpha_res']).relative_improvement_vs_maximal_pct.mean().agg(['min','max']).to_dict(),flush=True)
 # Replace the corresponding illustrations with both baselines, keeping panel labels.
 sweep=pd.read_csv(DATA/'comparisons_vs_maximal_sweep.csv');fig,axs=plt.subplots(1,2,figsize=(8,3.5),layout='constrained')
 for name,color in [('heuristic','#234f77'),('maximal','#319477'),('numerically_optimized','#d34b2c')]:
  d=prior[prior.policy==name];axs[0].plot(d.alpha_res,d.cum_incidence,'o-',color=color,label=name.replace('_',' ').capitalize());axs[1].plot(d.alpha_res,d.objective,'o-',color=color)
 no=prior[prior.policy=='no_control'];axs[0].plot(no.alpha_res,no.cum_incidence,':',color='#777777',label='No control');axs[1].plot(no.alpha_res,no.objective,':',color='#777777')
 axs[0].legend(fontsize=7);axs[0].set(ylabel='Cumulative new exposures',title='A. Epidemic burden');axs[1].set(ylabel='Objective J',title='B. Composite cost')
 for ax in axs:ax.set_xlabel('Resistance erosion rate (per day)');ax.grid(alpha=.2)
 fig.savefig(FIG/'fig2_alpha_sweep.pdf');plt.close(fig)
 aggregate=hd.groupby(['perturbation_q','alpha_res'])[['relative_improvement_vs_heuristic_pct','relative_improvement_vs_maximal_pct']].mean()
 fig,axs=plt.subplots(1,2,figsize=(10,3.2),layout='constrained',sharey=True)
 for ax,col,title in zip(axs,['relative_improvement_vs_heuristic_pct','relative_improvement_vs_maximal_pct'],['A. Compared with weak DFE heuristic','B. Compared with constant maximal']):
  d=aggregate[col].unstack();im=ax.imshow(d.values,aspect='auto',cmap='RdBu_r',vmin=-100,vmax=100);ax.set_xticks(range(6),[f'{x:.3f}' for x in d.columns],rotation=30);ax.set_yticks(range(3),[f'{x:.2f}' for x in d.index]);ax.set(title=title,xlabel='Resistance erosion rate (per day)')
  for i in range(3):
   for j in range(6):ax.text(j,i,f'{d.values[i,j]:.1f}',ha='center',va='center',fontsize=7)
 axs[0].set_ylabel('Parameter perturbation q');fig.colorbar(im,ax=axs,label='Mean objective saving (%)',shrink=.72)
 fig.savefig(FIG/'fig3_heatmap.pdf');plt.close(fig)
 oldsum=pd.read_csv(DATA/'uncertainty_summary.csv');fig,ax=plt.subplots(figsize=(7.5,3.7),layout='constrained')
 for d,mean,col,label in [(oldsum,'mean_relative_improvement','#234f77','Compared with weak DFE heuristic'),(sd,'mean_relative_improvement_vs_maximal_pct','#319477','Compared with constant maximal')]:
  ax.errorbar(d.alpha_res,d[mean],yerr=np.vstack([d[mean]-d.lower,d.upper-d[mean]]),fmt='o-',color=col,capsize=3,label=label)
 ax.axhline(0,color='black',ls='--',lw=.7);ax.set(xlabel='Resistance erosion rate (per day)',ylabel='Mean cost saving (%)');ax.legend(fontsize=8);fig.savefig(FIG/'fig6_uncertainty.pdf');plt.close(fig)
 (DATA/'minor_revision_manifest.json').write_text(json.dumps({'calculation':'Fixed original 12-interval policies and existing random draws for baseline comparisons; four-start optimization at w_V=0', 'numpy':np.__version__, 'scipy':'1.17.0','m':12,'seed_new_opt':203,'maxiter_new_opt':40,'same_perturbation_draws':True,'baseline_alpha':alpha},indent=2))

def optimize_viral_zero(p,old):
 uh,ah=heuristic(p);rng=np.random.default_rng(203)
 starts=[np.r_[uh/p.u_max,ah],np.ones(2*M),np.r_[np.zeros(M),np.ones(M)],rng.uniform(0,1,2*M),np.r_[old['u']/p.u_max,old['a']]]
 best=None;logs=[]
 def fun(x):return simulate(p,x[:M]*p.u_max,x[M:])['objective']/10000.
 for k,x0 in enumerate(starts):
  tic=time.time();res=minimize(fun,x0,method='L-BFGS-B',bounds=[(0,1)]*(2*M),options={'maxiter':40,'maxfun':3500,'ftol':1e-10,'gtol':1e-5,'eps':1e-5,'maxls':30})
  val=fun(res.x)*10000;rec={'start':k,'objective':val,'success':bool(res.success),'message':str(res.message),'nit':int(res.nit),'nfev':int(res.nfev),'seconds':time.time()-tic};logs.append(rec);print('OPT_WV0',rec,flush=True)
  if best is None or val<best[0]:best=(val,res.x)
 for x in starts:
  val=fun(x)*10000
  if val<best[0]:best=(val,x)
 return best[1][:M]*p.u_max,best[1][M:],logs

if __name__=='__main__':main()
