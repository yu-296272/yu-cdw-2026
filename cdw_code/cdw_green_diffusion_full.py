#!/usr/bin/env python3
"""CDW diffusion simulation and publication figures.

Models: RM (objective payoff), PT-MA (separate mental accounts), MA-HE
(maximum of integrated and separate prospect-theory evaluations).
Network: BA preferential attachment.
Updates: synchronous local rank-based Fermi rule.
"""
from __future__ import annotations
import argparse
import dataclasses
from dataclasses import dataclass, replace, asdict
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
import gzip

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from numba import njit, prange, set_num_threads

VERSION = "2.0.0"
MODEL_NAMES = ("RM", "PT-MA", "MA-HE")
MASTER_SEED = 4603053

@dataclass(frozen=True)
class Parameters:
    N: int = 100
    d: int = 4
    phi0: float = 0.30
    E: float = 10.0
    Ct: float = 6.0
    Cg0: float = 10.0
    K0: float = 6.0
    F0: float = 1.0
    delta: float = 0.35
    v: float = 0.50
    s: float = 1.50
    P: float = 1.0
    alpha: float = 0.88
    lam: float = 2.25
    kappa: float = 0.10
    T: int = 200
    steady_window: int = 20
    def validate(self) -> None:
        if self.N < 3 or self.d % 2 or not 2 <= self.d < 2*self.N:
            raise ValueError("N >= 3 and even target degree 2 <= d < 2N required")
        if not 0 <= self.phi0 <= 1 or self.T < self.steady_window or self.steady_window < 1:
            raise ValueError("Invalid initial proportion, horizon, or averaging window")
        if self.alpha <= 0 or self.lam <= 0 or self.kappa <= 0:
            raise ValueError("Positive alpha, lambda, and rank sensitivity required")
        if min(self.E,self.Ct,self.Cg0,self.K0,self.F0,self.s,self.P) < 0:
            raise ValueError("Revenue/cost/policy parameters cannot be negative")
    def vector(self) -> np.ndarray:
        return np.array([self.E,self.Ct,self.Cg0,self.K0,self.F0,self.delta,
                         self.v,self.s,self.P,self.alpha,self.lam,self.kappa],dtype=np.float64)

@njit(cache=True)
def value(x: float, alpha: float, lam: float) -> float:
    return x**alpha if x >= 0.0 else -lam*((-x)**alpha)

@njit(cache=True)
def simulate_one(neighbors, degree, permutation, neighbor_u, imitate_u,
                 initial_count, p, model, horizon):
    n = degree.size
    E,Ct,Cg0,K0,F0,delta,v,s,P,alpha,lam,kappa = p
    cgk = (Cg0 + K0)*math.exp(-delta)
    A = E-Ct-P
    state = np.zeros(n, dtype=np.int8)
    for q in range(initial_count):
        state[permutation[q]] = 1
    rm = np.full(n,E)
    rt = np.full(n,Ct+P)
    M = np.empty(n)
    C = np.empty(n)
    utility = np.empty(n)
    ranks = np.empty(n)
    nxt = np.empty(n,dtype=np.int8)
    phi_path = np.empty(horizon+1)
    payoff_path = np.empty(horizon+1)
    for t in range(horizon+1):
        count = 0
        for i in range(n):
            count += state[i]
        phi = count/n
        mg = (1.0+v)*E+s*(1.0-phi)
        cg = cgk+F0*math.exp(-phi)
        B = mg-cg
        phi_path[t] = phi
        payoff_path[t] = (1.0-phi)*A+phi*B
        if count == 0 or count == n:
            for q in range(t+1,horizon+1):
                phi_path[q] = phi
                payoff_path[q] = payoff_path[t]
            break
        if t == horizon:
            break
        for i in range(n):
            M[i] = mg if state[i] else E
            C[i] = cg if state[i] else Ct+P
            if model == 0:
                utility[i] = M[i]-C[i]
            else:
                dm = M[i]-rm[i]
                dc = C[i]-rt[i]
                sep = value(dm,alpha,lam)+value(-dc,alpha,lam)
                if model == 1:
                    utility[i] = sep
                else:
                    integ = value(dm-dc,alpha,lam)
                    utility[i] = max(integ,sep)
        for i in range(n):
            r = 0.0
            for q in range(degree[i]):
                k = neighbors[i,q]
                if utility[k] < utility[i]:
                    r += 1.0
                elif utility[k] == utility[i]:
                    r += 0.5
            ranks[i] = r/degree[i]
        for i in range(n):
            q = int(neighbor_u[t,i]*degree[i])
            j = neighbors[i,q]
            prob = 1.0/(1.0+math.exp(-(ranks[j]-ranks[i])/kappa))
            nxt[i] = state[j] if imitate_u[t,i] < prob else state[i]
        for i in range(n):
            rm[i] = M[i]
            rt[i] = C[i]
            state[i] = nxt[i]
    return phi_path,payoff_path

@njit(cache=True,parallel=True)
def ensemble(neighbors,degree,permutations,neighbor_u,imitate_u,initial_count,p,T,window):
    reps = permutations.shape[0]
    terminal = np.empty((reps,3,3))
    path = np.empty((reps,3,2,T+1))
    for r in prange(reps):
        for model in range(3):
            ph,pi = simulate_one(neighbors[r],degree[r],permutations[r],
                                neighbor_u[r],imitate_u[r],initial_count,p,model,T)
            terminal[r,model,0] = np.mean(ph[T-window+1:T+1])
            terminal[r,model,1] = np.mean(pi[T-window+1:T+1])
            terminal[r,model,2] = ph[T]
            path[r,model,0,:] = ph
            path[r,model,1,:] = pi
    return terminal,path

def make_ba(n: int, m: int, seed: int):
    rng = np.random.Generator(np.random.PCG64(seed))
    adjacent = [set() for _ in range(n)]
    repeated = []
    for j in range(1,m+1):
        adjacent[0].add(j); adjacent[j].add(0)
    for i in range(m+1):
        repeated.extend([i]*len(adjacent[i]))
    for new in range(m+1,n):
        targets = set()
        while len(targets)<m:
            targets.add(int(repeated[int(rng.integers(len(repeated)))]))
        for j in sorted(targets):
            adjacent[new].add(j); adjacent[j].add(new)
            repeated.append(j)
        repeated.extend([new]*m)
    degree = np.array([len(a) for a in adjacent],dtype=np.int16)
    neighbors = np.full((n,n),-1,dtype=np.int16)
    for i,a in enumerate(adjacent):
        neighbors[i,:len(a)] = sorted(a)
    return neighbors,degree

def seed_for(master: int, replication: int, stream: int, d: int=0) -> int:
    return int(np.random.SeedSequence([master,replication,stream,d]).generate_state(1)[0])

class RandomInputs:
    def __init__(self, params: Parameters, replications: int, seed: int):
        self.p,self.reps,self.seed = params,replications,seed
        self.perm = np.empty((replications,params.N),dtype=np.int16)
        self.nu = np.empty((replications,params.T,params.N),dtype=np.float64)
        self.iu = np.empty_like(self.nu)
        self.graphs = {}
        self.seeds = []
        for r in range(replications):
            si,sn,su = (seed_for(seed,r,s) for s in (1,2,3))
            self.perm[r] = np.random.Generator(np.random.PCG64(si)).permutation(params.N)
            self.nu[r] = np.random.Generator(np.random.PCG64(sn)).random((params.T,params.N))
            self.iu[r] = np.random.Generator(np.random.PCG64(su)).random((params.T,params.N))
            self.seeds.append(dict(replication=r,initialization_seed=si,neighbor_seed=sn,imitation_seed=su))
    def graph(self,d):
        if d not in self.graphs:
            g = [make_ba(self.p.N,d//2,seed_for(self.seed,r,0,d)) for r in range(self.reps)]
            self.graphs[d] = (np.stack([a for a,b in g]),np.stack([b for a,b in g]))
        return self.graphs[d]


def scenario_id(p: Parameters,reps: int,seed: int) -> str:
    blob = json.dumps(dict(version=VERSION,parameters=asdict(p),replications=reps,seed=seed),sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]

def mean_ci(x: np.ndarray):
    x = np.asarray(x,dtype=float)
    mean = x.mean(axis=0)
    half = student_t.ppf(.975,len(x)-1)*x.std(axis=0,ddof=1)/math.sqrt(len(x))
    return mean,mean-half,mean+half

def objective(phi,p: Parameters):
    phi = np.asarray(phi)
    A = p.E-p.Ct-p.P
    D = p.v*p.E+p.Ct+p.P-(p.Cg0+p.K0)*np.exp(-p.delta)-p.F0*np.exp(-phi)+p.s*(1-phi)
    return A+phi*D

def scenario_plan():
    b = Parameters()
    plan = []
    def add(group,label,p,**extra):
        plan.append(dict(group=group,label=label,p=p,**extra))
    add('baseline','baseline',b)
    for x in np.round(np.arange(.1,.90001,.05),8):
        add('one_parameter','delta',replace(b,delta=float(x)),x=float(x),parameter='delta')
    for x in np.round(np.arange(.2,.80001,.05),8):
        add('one_parameter','v',replace(b,v=float(x)),x=float(x),parameter='v')
    for d in np.round(np.arange(.1,.90001,.025),8):
        for v in np.round(np.arange(.2,.80001,.025),8):
            add('model_difference','delta_v',replace(b,delta=float(d),v=float(v)),x=float(d),y=float(v))
    for d in np.round(np.arange(.1,.90001,.05),8):
        for l in np.round(np.arange(1,4.0001,.25),8):
            add('behavior','delta_lambda',replace(b,delta=float(d),lam=float(l)),x=float(d),y=float(l))
    for ph in np.round(np.arange(.05,.65001,.05),8):
        for d in (2,4,6,8,10,12):
            add('network','phi0_d',replace(b,phi0=float(ph),d=d),x=float(ph),y=d)
    for s in np.round(np.arange(0,4.0001,.25),8):
        for pen in np.round(np.arange(0,3.0001,.25),8):
            add('policy','s_P',replace(b,phi0=.2,s=float(s),P=float(pen)),x=float(s),y=float(pen))
    add('sensitivity','baseline',b,parameter='baseline',factor=1.)
    for k in ('Ct','Cg0','K0','F0','s','P'):
        for factor in (.8,1.2):
            add('sensitivity',f'{k}_{factor:g}',replace(b,**{k:getattr(b,k)*factor}),parameter=k,factor=factor)
    for s in np.arange(0,8.0001,1.0):
        for pen in np.arange(0,4.0001,1.0):
            if s>4 or pen>3:
                add('outer_policy_check','s_P_outer',replace(b,phi0=.2,s=float(s),P=float(pen)),x=float(s),y=float(pen))
    return plan

def run_all(out: Path,reps=500,seed=MASTER_SEED,jobs=4,smoke=False):
    out.mkdir(parents=True,exist_ok=True)
    raw = out/'data'/'raw'; raw.mkdir(parents=True,exist_ok=True)
    set_num_threads(jobs)
    inputs = RandomInputs(Parameters(),reps,seed)
    plan = scenario_plan()
    if smoke:
        plan = [a for a in plan if a['group']=='baseline' or (a['group']=='model_difference' and a['x']==.35 and a['y']==.45)]
    unique = {}
    for a in plan:
        a['id'] = scenario_id(a['p'],reps,seed)
        unique[a['id']] = a['p']
    start = time.time()
    base_id = scenario_id(Parameters(),reps,seed)
    for num,(sid,p) in enumerate(unique.items(),1):
        f = raw/f'{sid}.npz'
        if not f.exists():
            p.validate()
            nb,deg = inputs.graph(p.d)
            terminal,path = ensemble(nb,deg,inputs.perm,inputs.nu,inputs.iu,int(round(p.N*p.phi0)),p.vector(),p.T,p.steady_window)
            np.savez_compressed(f,terminal=terminal,parameters=json.dumps(asdict(p)),scenario_id=sid)
            if sid==base_id:
                np.savez_compressed(out/'data'/'baseline_paths.npz',paths=path)
        if num%25==0 or num==1 or num==len(unique):
            print(f'{num}/{len(unique)} parameter combinations, elapsed {time.time()-start:.1f}s',flush=True)
    pd.DataFrame(inputs.seeds).to_csv(out/'data'/'replication_seeds.csv',index=False)
    gseed = [dict(replication=r,d=d,network_seed=seed_for(seed,r,0,d),realized_mean_degree=d*(Parameters().N-d//2)/Parameters().N)
             for d in (2,4,6,8,10,12) for r in range(reps)]
    pd.DataFrame(gseed).to_csv(out/'data'/'network_seeds.csv',index=False)
    manifest = dict(version=VERSION,master_seed=seed,replications=reps,
                    baseline=asdict(Parameters()),scenario_count=len(unique),plan_count=len(plan),
                    random_generator='numpy.random.Generator(PCG64), SeedSequence([master,replication,stream,d])',
                    pairing='Identical graph, initial allocation and neighbor/imitation draws across models; independent replication indices.',
                    averaging_rounds='181-200 inclusive; t=0 is initialization; all mixed runs retained',
                    confidence='Two-sided Student t intervals for means; paired differences for model comparisons; pointwise, not simultaneous.',
                    python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'data'/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    serial = [{**{k:v for k,v in a.items() if k!='p'},'parameters':asdict(a['p'])} for a in plan]
    (out/'data'/'scenario_plan.json').write_text(json.dumps(serial,indent=2),encoding='utf-8')
    summarize(out)


def summarize(out: Path):
    plan=json.loads((out/'data'/'scenario_plan.json').read_text())
    rows=[]; comparisons=[]
    seen=set(); rawrows=[]
    for a in plan:
        p=Parameters(**a['parameters'])
        z=np.load(out/'data'/'raw'/f"{a['id']}.npz")['terminal']
        meta={k:v for k,v in a.items() if k not in ('parameters','id')}
        for m,name in enumerate(MODEL_NAMES):
            pm,pl,ph=mean_ci(z[:,m,0]); ym,yl,yh=mean_ci(z[:,m,1])
            rows.append(dict(**meta,scenario_id=a['id'],model=name,phi_mean=pm,phi_ci_low=pl,phi_ci_high=ph,
                             payoff_mean=ym,payoff_ci_low=yl,payoff_ci_high=yh,
                             final_all_green=int(np.sum(z[:,m,2]==1)),final_all_conventional=int(np.sum(z[:,m,2]==0)),
                             final_mixed=int(np.sum((z[:,m,2]>0)&(z[:,m,2]<1))),**asdict(p)))
        if a['group']=='model_difference':
            row=dict(scenario_id=a['id'],delta=p.delta,v=p.v)
            for left,right,tag in ((2,0,'he_rm'),(2,1,'he_pt')):
                for j,metric in enumerate(('phi','payoff')):
                    mu,lo,hi=mean_ci(z[:,left,j]-z[:,right,j])
                    row.update({f'{tag}_{metric}_mean':mu,f'{tag}_{metric}_ci_low':lo,f'{tag}_{metric}_ci_high':hi,
                                f'{tag}_{metric}_region':1 if lo>0 else (-1 if hi<0 else 0)})
            comparisons.append(row)
        if a['id'] not in seen:
            seen.add(a['id'])
            for r in range(z.shape[0]):
                for m,name in enumerate(MODEL_NAMES):
                    rawrows.append((a['id'],r,name,z[r,m,0],z[r,m,1],z[r,m,2]))
    df=pd.DataFrame(rows)
    df.to_csv(out/'data'/'all_summary.csv',index=False)
    for group in df.group.unique():
        df[df.group==group].to_csv(out/'data'/f'{group}_summary.csv',index=False)
    cp=pd.DataFrame(comparisons)
    cp.to_csv(out/'data'/'model_differences.csv',index=False)
    if len(cp):
        cp[(np.isclose(cp.delta,.35))&(np.isclose(cp.v,.45))].to_csv(out/'data'/'representative_differences.csv',index=False)
    pd.DataFrame(rawrows,columns=['scenario_id','replication','model','phi_terminal','payoff_terminal','phi_at_T']).to_csv(
        out/'data'/'replication_results.csv.gz',index=False,compression='gzip')
    bs=out/'data'/'baseline_paths.npz'
    if bs.exists():
        paths=np.load(bs)['paths']; br=[]
        for m,name in enumerate(MODEL_NAMES):
            for j,metric in enumerate(('phi','payoff')):
                mean,lo,hi=mean_ci(paths[:,m,j,:])
                for t in range(paths.shape[-1]):
                    br.append(dict(model=name,metric=metric,t=t,mean=mean[t],ci_low=lo[t],ci_high=hi[t]))
        pd.DataFrame(br).to_csv(out/'data'/'baseline_trajectories.csv',index=False)
    sensitivity=df[df.group=='sensitivity'].copy()
    sensitivity['D_initial']=[float(objective(p.phi0,p)-(p.E-p.Ct-p.P))/p.phi0 for p in (Parameters(**{k:r[k] for k in asdict(Parameters())}) for _,r in sensitivity.iterrows())]
    sensitivity['D_full']=[float(objective(1,p)-(p.E-p.Ct-p.P)) for p in (Parameters(**{k:r[k] for k in asdict(Parameters())}) for _,r in sensitivity.iterrows())]
    sensitivity.to_csv(out/'data'/'Table_S1.csv',index=False)


def self_test():
    import tempfile
    p=Parameters()
    assert abs(objective(0,p)-3)<1e-12
    expected=15-16*math.exp(-.35)-math.exp(-1)
    assert abs(objective(1,p)-expected)<1e-12
    assert value(-2,.88,2.25)<0 and value(2,.88,2.25)>0
    for dm in (-5.,0.,3.):
        for dc in (-2.,0.,4.):
            sep=value(dm,.88,2.25)+value(-dc,.88,2.25)
            he=max(sep,value(dm-dc,.88,2.25))
            assert he>=sep
    for d in (2,4,6,8,10,12):
        nb,deg=make_ba(100,d//2,123)
        assert int(deg.sum())==d*(100-d//2)
        for i in range(100):
            assert i not in nb[i,:deg[i]]
            assert len(set(nb[i,:deg[i]]))==deg[i]
            for j in nb[i,:deg[i]]:
                assert i in nb[j,:deg[j]]
    inputs=RandomInputs(p,3,123)
    nb,deg=inputs.graph(4)
    a,path=ensemble(nb,deg,inputs.perm,inputs.nu,inputs.iu,30,p.vector(),p.T,p.steady_window)
    b,path2=ensemble(nb,deg,inputs.perm,inputs.nu,inputs.iu,30,p.vector(),p.T,p.steady_window)
    assert np.array_equal(a,b) and np.array_equal(path,path2)
    assert np.allclose(path[:,:,1,:],objective(path[:,:,0,:],p),rtol=1e-12,atol=1e-12)
    assert np.allclose(a[:,:,0],path[:,:,0,181:201].mean(axis=2))
    linear=replace(p,alpha=1.,lam=1.)
    z,pa=ensemble(nb,deg,inputs.perm,inputs.nu,inputs.iu,30,linear.vector(),p.T,p.steady_window)
    assert np.array_equal(pa[:,1],pa[:,2])
    print('All implementation tests passed.',flush=True)

def plot_all(out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,Normalize
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter,MaxNLocator
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    fdir=out/'figures'; fdir.mkdir(exist_ok=True)
    df=pd.read_csv(out/'data'/'all_summary.csv')
    cp=pd.read_csv(out/'data'/'model_differences.csv')
    bt=pd.read_csv(out/'data'/'baseline_trajectories.csv')
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
                         'mathtext.fontset':'dejavuserif','font.size':11,
                         'axes.labelsize':12,'xtick.labelsize':10.5,'ytick.labelsize':10.5,
                         'legend.fontsize':10,'axes.linewidth':.9,'savefig.dpi':300,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'hatch.linewidth':.55})
    colors={'RM':'#1f77b4','PT-MA':'#ff7f0e','MA-HE':'#2ca02c'}
    lines={'RM':'-','PT-MA':'--','MA-HE':'-.'}
    markers={'RM':'o','PT-MA':'s','MA-HE':'^'}
    sample=np.array([[157,204,229],[187,219,236],[220,237,243],[240,245,244],
                     [247,246,238],[255,247,231],[251,222,204],[246,197,175],
                     [240,177,157],[232,159,146],[225,144,135]])/255.
    cmap=LinearSegmentedColormap.from_list('reference_blue_cream_coral',sample,N=256)
    fmt=FuncFormatter(lambda x,pos:f'{x:g}')
    pdf=PdfPages(fdir/'All_main_figures.pdf')
    def save(fig,name,main=True):
        for extension in ('png','pdf','svg'):
            fig.savefig(fdir/f'{name}.{extension}',bbox_inches='tight')
        if main: pdf.savefig(fig,bbox_inches='tight')
        plt.close(fig)
    def axes_style(ax):
        ax.grid(True,color='#d7d7d7',linewidth=.6,alpha=.9)
        ax.set_axisbelow(True); ax.xaxis.set_major_formatter(fmt);ax.yaxis.set_major_formatter(fmt)
    def panel(ax,label):
        ax.text(.5,-.25,f'({label})',ha='center',va='top',transform=ax.transAxes,fontsize=12)
    def curves(ax,data,metric,xcol,marked=False):
        for model in MODEL_NAMES:
            d=data[data.model==model].sort_values(xcol)
            if d.empty: continue
            x=d[xcol].to_numpy(); y=d[f'{metric}_mean'].to_numpy()
            lo=d[f'{metric}_ci_low'].to_numpy(); hi=d[f'{metric}_ci_high'].to_numpy()
            if metric=='phi':lo=np.clip(lo,0,1);hi=np.clip(hi,0,1)
            ax.fill_between(x,lo,hi,color=('#e7aabd' if model=='PT-MA' else colors[model]),alpha=.25,linewidth=0,zorder=1)
            ax.plot(x,y,color=colors[model],linestyle=lines[model],linewidth=1.7,
                    marker=markers[model] if marked else None,markersize=4.7,
                    markerfacecolor='white',markeredgewidth=1,label=model,zorder=3)
    fig,axs=plt.subplots(1,2,figsize=(10.2,4.1))
    for ax,metric,ylabel,lab in zip(axs,('phi','payoff'),
          (r'Adoption proportion ($\varphi$)',r'Average objective net payoff ($\overline{\Pi}$)'),('a','b')):
        for model in MODEL_NAMES:
            d=bt[(bt.model==model)&(bt.metric==metric)]
            lo=d.ci_low.to_numpy();hi=d.ci_high.to_numpy()
            if metric=='phi':lo=np.clip(lo,0,1);hi=np.clip(hi,0,1)
            ax.fill_between(d.t,lo,hi,color=('#e7aabd' if model=='PT-MA' else colors[model]),alpha=.23,linewidth=0)
            ax.plot(d.t,d['mean'],color=colors[model],linestyle=lines[model],linewidth=1.8,label=model)
        ax.set_xlabel('Evolutionary round ($t$)');ax.set_ylabel(ylabel)
        ax.set_xlim(0,200);axes_style(ax);panel(ax,lab)
        if metric=='phi':ax.set_ylim(-.02,1.02)
        ax.legend(loc='best',framealpha=.95)
    fig.subplots_adjust(left=.085,right=.985,top=.96,bottom=.23,wspace=.31)
    save(fig,'fig01_baseline')
    fig,axs=plt.subplots(1,2,figsize=(10.4,5.1))
    xs=np.sort(cp.delta.unique());ys=np.sort(cp.v.unique());X,Y=np.meshgrid(xs,ys)
    for ax,metric,lab in zip(axs,('phi','payoff'),('a','b')):
        lo=cp.pivot(index='v',columns='delta',values=f'he_rm_{metric}_ci_low').reindex(index=ys,columns=xs).to_numpy()
        hi=cp.pivot(index='v',columns='delta',values=f'he_rm_{metric}_ci_high').reindex(index=ys,columns=xs).to_numpy()
        eps=1e-12
        for field,hatch in ((lo-eps,'////'),(-hi-eps,'\\\\\\\\')):
            if np.max(field)>0:
                ax.contourf(X,Y,field,levels=[0.,float(np.max(field)+1.)],colors='none',hatches=[hatch])
                if np.min(field)<0:
                    ax.contour(X,Y,field,levels=[0.],colors='black',linewidths=.65)
        ax.set_xlabel(r'Technological maturity ($\delta$)')
        ax.set_ylabel(r'Green revenue premium ($\nu$)')
        ax.set_xlim(.1,.9);ax.set_ylim(.2,.8);ax.set_box_aspect(1)
        ax.set_xticks([.1,.3,.5,.7,.9]);ax.set_yticks([.2,.3,.4,.5,.6,.7,.8])
        ax.xaxis.set_major_formatter(fmt);ax.yaxis.set_major_formatter(fmt)
        legend=[Patch(facecolor='white',edgecolor='black',hatch='////',label='MA-HE > RM'),
                Patch(facecolor='white',edgecolor='black',hatch='\\\\\\\\',label='MA-HE < RM'),
                Patch(facecolor='white',edgecolor='black',label='Not distinguished')]
        ax.legend(handles=legend,loc='upper right',framealpha=1,fontsize=9.3,
                  borderpad=.65,labelspacing=.5,handlelength=2)
        panel(ax,lab)
    fig.subplots_adjust(left=.075,right=.99,top=.985,bottom=.18,wspace=.25)
    save(fig,'fig02_model_difference')
    one=df[df.group=='one_parameter']
    for metric,fname,ylabel in [('phi','fig03_one_parameter_phi',r'Terminal adoption proportion ($\varphi^*$)'),
                                ('payoff','fig04_one_parameter_profit',r'Average objective net payoff ($\overline{\Pi}^{*}$)')]:
        fig,axs=plt.subplots(1,2,figsize=(10.2,4.1))
        for ax,param,xlabel,lab in zip(axs,('delta','v'),(r'Technological maturity ($\delta$)',r'Green revenue premium ($\nu$)'),('a','b')):
            curves(ax,one[one.parameter==param],metric,'x',True)
            ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);axes_style(ax);panel(ax,lab)
            ax.legend(loc='best',framealpha=.95)
            if metric=='phi':ax.set_ylim(-.03,1.04)
        fig.subplots_adjust(left=.085,right=.985,top=.96,bottom=.23,wspace=.31)
        save(fig,fname)
    for group,fname,xlabel,ylabel in [
        ('behavior','fig05_behavior_heatmap',r'Technological maturity ($\delta$)',r'Loss aversion ($\lambda$)'),
        ('network','fig06_network_heatmap',r'Initial adoption proportion ($\varphi_0$)',r'Target mean degree ($d$)'),
        ('policy','fig07_policy_heatmap',r'Subsidy intensity ($s$)',r'Environmental penalty ($P$)')]:
        sub=df[(df.group==group)&df.model.isin(['PT-MA','MA-HE'])]
        fig,axs=plt.subplots(2,2,figsize=(9.8,8.0))
        x=np.sort(sub.x.unique());y=np.sort(sub.y.unique())
        for row,metric in enumerate(('phi','payoff')):
            values=sub[f'{metric}_mean']
            norm=Normalize(0,1) if metric=='phi' else Normalize(float(values.min()),float(values.max()))
            for col,model in enumerate(('PT-MA','MA-HE')):
                ax=axs[row,col];sm=sub[sub.model==model]
                z=sm.pivot(index='y',columns='x',values=f'{metric}_mean').reindex(index=y,columns=x).to_numpy()
                im=ax.pcolormesh(x,y,z,shading='nearest',cmap=cmap,norm=norm,rasterized=False)
                ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
                ax.set_xlim(x.min(),x.max());ax.set_ylim(y.min(),y.max())
                if row==0:ax.set_title(model,pad=10,fontsize=12)
                ax.xaxis.set_major_formatter(fmt);ax.yaxis.set_major_formatter(fmt)
                if group=='network':ax.set_yticks(y);ax.set_xticks([.05,.2,.35,.5,.65])
                elif group=='behavior':ax.set_xticks([.1,.3,.5,.7,.9]);ax.set_yticks([1,2,3,4])
                else:ax.set_xticks([0,1,2,3,4]);ax.set_yticks([0,1,2,3])
                ax.text(.5,-.28,f'({chr(97+row*2+col)})',ha='center',va='top',transform=ax.transAxes,fontsize=12)
            cax=fig.add_axes([.88,.57 if row==0 else .11,.019,.31])
            cb=fig.colorbar(im,cax=cax)
            cb.set_label(r'Adoption proportion ($\varphi^*$)' if metric=='phi' else r'Objective net payoff ($\overline{\Pi}^{*}$)',fontsize=11,labelpad=9)
            cb.ax.yaxis.set_major_formatter(fmt)
        fig.subplots_adjust(left=.09,right=.85,top=.92,bottom=.105,hspace=.59,wspace=.34)
        save(fig,fname)
    pdf.close()
    sl=df[(df.group=='model_difference')&np.isclose(df.delta,.35)]
    fig,axs=plt.subplots(1,2,figsize=(10.2,4.1))
    for ax,metric,ylabel,lab in zip(axs,('phi','payoff'),
                (r'Terminal adoption proportion ($\varphi^*$)',r'Average objective net payoff ($\overline{\Pi}^{*}$)'),('a','b')):
        curves(ax,sl,metric,'v',True)
        ax.set_xlabel(r'Green revenue premium ($\nu$)');ax.set_ylabel(ylabel)
        axes_style(ax);panel(ax,lab);ax.legend(loc='best',framealpha=.95)
        if metric=='phi':ax.set_ylim(-.03,1.04)
    fig.subplots_adjust(left=.085,right=.985,top=.96,bottom=.23,wspace=.31)
    save(fig,'Figure_S1_transition_slice',False)
    print('Seven main figures and Figure S1 exported (PNG/PDF/SVG).',flush=True)


def verify_outputs(out: Path):
    manifest=json.loads((out/'data'/'run_manifest.json').read_text())
    df=pd.read_csv(out/'data'/'all_summary.csv')
    assert not df[['phi_mean','payoff_mean']].isna().any().any()
    assert df.phi_mean.between(-1e-12,1+1e-12).all()
    assert (df.final_all_green+df.final_all_conventional+df.final_mixed==manifest['replications']).all()
    assert len(df[df.group=='sensitivity'])==39
    assert 'fig08' not in ' '.join(p.name for p in (out/'figures').glob('*'))
    for (_,model),g in df.groupby(['scenario_id','model']):
        assert g.phi_mean.nunique()==1 and g.payoff_mean.nunique()==1
    print('Output integrity checks passed.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out',type=Path,default=Path('.'))
    parser.add_argument('--replications',type=int,default=500)
    parser.add_argument('--seed',type=int,default=MASTER_SEED)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    if args.test_only:
        set_num_threads(max(1,args.jobs));self_test();return
    if args.replications<2:parser.error('At least two replications are needed for intervals')
    if args.plots_only:
        plot_all(args.out);verify_outputs(args.out);return
    if args.summarize_only:
        summarize(args.out);return
    run_all(args.out,args.replications,args.seed,args.jobs,args.smoke_test)
    if not args.smoke_test:
        plot_all(args.out);verify_outputs(args.out)

if __name__=='__main__':
    main()