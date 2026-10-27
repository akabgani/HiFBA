#!/usr/bin/env python3
import argparse, json, math, csv
from pathlib import Path
import numpy as np

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_jsons(path):
    p=Path(path)
    files=[p] if p.is_file() else sorted(p.rglob('*.json'))
    data={}
    for fp in files:
        obj=load_json(fp); data.update(obj.get('user_data', {}))
    users=sorted(data.keys()); out={}
    for u in users:
        out[u]=(np.asarray(data[u]['x'], dtype=np.float64), np.asarray(data[u]['y'], dtype=np.int64))
    return users,out

def softmax(z):
    z=z-z.max(axis=1, keepdims=True); e=np.exp(z); return e/e.sum(axis=1, keepdims=True)

def loss_acc(theta, X, y, d, c, batch=4096):
    W=theta[:d*c].reshape(d,c); b=theta[d*c:]; n=len(y); loss=0.0; correct=0
    for s in range(0,n,batch):
        xb=X[s:s+batch]; yb=y[s:s+batch]; z=xb@W+b; p=softmax(z)
        loss += -np.log(np.maximum(p[np.arange(len(yb)), yb],1e-300)).sum(); correct += (np.argmax(z,axis=1)==yb).sum()
    return loss/max(n,1), correct/max(n,1)

def grad_ce(theta, X, y, d, c):
    W=theta[:d*c].reshape(d,c); b=theta[d*c:]; p=softmax(X@W+b)
    p[np.arange(len(y)), y]-=1.0; p/=len(y)
    return np.concatenate([(X.T@p).ravel(),p.sum(axis=0)])

def power_grad(delta, mu, p, R):
    if mu==0: return np.zeros_like(delta)
    if p==2: return mu*delta
    nrm=float(np.linalg.norm(delta))
    if nrm==0: return np.zeros_like(delta)
    return mu*(R**(2.0-p))*(nrm**(p-2.0))*delta

def local_train(global_theta, X, y, epochs, batch_size, lr, mu, p, R, d, c):
    theta=global_theta.copy(); n=len(y); order=np.arange(n); fixed_perm=np.random.RandomState(100).permutation(n)
    for _ in range(int(epochs)):
        order=order[fixed_perm]
        for s in range(0,n,batch_size):
            idx=order[s:s+batch_size]
            if len(idx)==0: continue
            g=grad_ce(theta,X[idx],y[idx],d,c)+power_grad(theta-global_theta,mu,p,R)
            theta-=lr*g
    return theta

def init_theta(d,c,seed):
    rng=np.random.RandomState(123+seed); lim=math.sqrt(6.0/(d+c))
    return np.concatenate([rng.uniform(-lim,lim,size=(d,c)).ravel(),np.zeros(c)])

def evaluate_all(theta,data,d,c):
    total_n=0; total_loss=0.0; total_correct=0.0
    for X,y in data.values():
        l,a=loss_acc(theta,X,y,d,c); total_loss+=l*len(y); total_correct+=a*len(y); total_n+=len(y)
    return total_loss/max(total_n,1), total_correct/max(total_n,1)

def make_round_schedule(nclients,rounds,cpr,drop,E):
    schedules=[]
    for r in range(rounds):
        rs=np.random.RandomState(r); selected=rs.choice(np.arange(nclients),min(cpr,nclients),replace=False)
        n_active=round(len(selected)*(1-drop)); active=set(rs.choice(selected,n_active,replace=False).tolist()) if n_active>0 else set(); epochs={}
        for ci in selected.tolist(): epochs[ci]=E if ci in active else int(rs.randint(1,E))
        schedules.append((selected.tolist(),active,epochs))
    return schedules

def run_method(users,train,test,method,drop,rounds,cpr,E,batch,lr,mu,p,R,seed,num_classes,eval_every=1):
    d=train[users[0]][0].shape[1]; c=num_classes; theta=init_theta(d,c,seed); schedules=make_round_schedule(len(users),rounds,cpr,drop,E)
    history=[]; all_update_norms=[]
    for r,(sel,active,epochs) in enumerate(schedules):
        locals_=[]; weights=[]; norms=[]; use=active if method=='fedavg' else set(sel)
        for ci in sel:
            if ci not in use: continue
            u=users[ci]; X,y=train[u]
            if method=='fedavg': e=E; lmu=0.0; lp=2.0
            elif method=='fedprox': e=epochs[ci]; lmu=mu; lp=2.0
            elif method=='hifedprox': e=epochs[ci]; lmu=mu; lp=p
            else: raise ValueError(method)
            th=local_train(theta,X,y,e,batch,lr,lmu,lp,R,d,c); locals_.append(th); weights.append(len(y)); norms.append(float(np.linalg.norm(th-theta)))
        if locals_:
            w=np.asarray(weights,dtype=np.float64); w/=w.sum(); theta=np.sum(np.stack(locals_)*w[:,None],axis=0)
        all_update_norms.extend(norms)
        if r%eval_every==0 or r==rounds-1:
            tr_l,tr_a=evaluate_all(theta,train,d,c); te_l,te_a=evaluate_all(theta,test,d,c)
            q50=float(np.quantile(norms,.5)) if norms else float('nan'); q90=float(np.quantile(norms,.9)) if norms else float('nan')
            history.append({'round':r+1,'train_loss':tr_l,'train_acc':tr_a,'test_loss':te_l,'test_acc':te_a,'update_q50':q50,'update_q90':q90,'tail_q90_q50':q90/max(q50,1e-15)})
            print(f'{method} drop={drop:.1f} seed={seed} r={r+1}: test_acc={te_a:.4f} train_loss={tr_l:.4f}',flush=True)
    return {'method':method,'drop':drop,'seed':seed,'p':p if method=='hifedprox' else (2 if method=='fedprox' else None),'mu':mu if method!='fedavg' else 0.0,'R':R,'final':history[-1],'update_norm_q90_all':float(np.quantile(all_update_norms,.9)) if all_update_norms else None,'update_norm_q50_all':float(np.quantile(all_update_norms,.5)) if all_update_norms else None,'history':history}

def dataset_stats(users,train,test):
    sizes=np.array([len(train[u][1])+len(test.get(u,(None,[]))[1]) for u in users]); labels=sorted(set(int(v) for u in users for v in np.unique(train[u][1]))); classes_per=[len(np.unique(train[u][1])) for u in users]
    return {'clients':len(users),'total_samples':int(sizes.sum()),'train_samples':int(sum(len(train[u][1]) for u in users)),'test_samples':int(sum(len(test[u][1]) for u in users if u in test)),'samples_per_client_mean':float(sizes.mean()),'samples_per_client_std':float(sizes.std()),'labels':labels,'classes_per_client_min':int(min(classes_per)),'classes_per_client_median':float(np.median(classes_per)),'classes_per_client_max':int(max(classes_per))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',required=True); ap.add_argument('--test',required=True); ap.add_argument('--out',default='results')
    ap.add_argument('--rounds',type=int,default=200); ap.add_argument('--clients-per-round',type=int,default=10); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=10)
    ap.add_argument('--lr',type=float,default=0.003); ap.add_argument('--mu',type=float,default=1.0); ap.add_argument('--power',type=float,default=6.0); ap.add_argument('--num-classes',type=int,default=26)
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); ap.add_argument('--drops',type=float,nargs='+',default=[0,0.5,0.9]); ap.add_argument('--methods',nargs='+',choices=['fedavg','fedprox','hifedprox'],default=['fedavg','fedprox','hifedprox']); ap.add_argument('--pilot-rounds',type=int,default=20); ap.add_argument('--eval-every',type=int,default=5); args=ap.parse_args()
    users,tr=merge_jsons(args.train); _,te=merge_jsons(args.test); users=[u for u in users if u in te]; stats=dataset_stats(users,tr,te); print('DATASET_STATS',json.dumps(stats),flush=True)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); (out/'dataset_stats.json').write_text(json.dumps(stats,indent=2))
    pilot=run_method(users,tr,te,'fedprox',0.0,args.pilot_rounds,args.clients_per_round,args.epochs,args.batch_size,args.lr,args.mu,2.0,1.0,0,args.num_classes,max(args.pilot_rounds,1)); R=max(float(pilot['update_norm_q90_all']),1e-8); print('CALIBRATED_R',R,flush=True)
    results=[]
    for seed in args.seeds:
        for drop in args.drops:
            for method in args.methods:
                res=run_method(users,tr,te,method,drop,args.rounds,args.clients_per_round,args.epochs,args.batch_size,args.lr,args.mu,args.power,R,seed,args.num_classes,args.eval_every); results.append(res); (out/f'{method}_drop{drop:g}_seed{seed}.json').write_text(json.dumps(res,indent=2))
    rows=[]
    for res in results:
        f=res['final']; rows.append({'method':res['method'],'drop':res['drop'],'seed':res['seed'],'p':res['p'],'mu':res['mu'],'R':R,'test_acc':f['test_acc'],'test_loss':f['test_loss'],'train_acc':f['train_acc'],'train_loss':f['train_loss'],'tail_q90_q50':f['tail_q90_q50'],'update_q90_all':res['update_norm_q90_all'],'update_q50_all':res['update_norm_q50_all']})
    with open(out/'summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    agg=[]
    for drop in args.drops:
        for method in args.methods:
            rr=[x for x in rows if x['drop']==drop and x['method']==method]; a=np.array([x['test_acc'] for x in rr]); l=np.array([x['test_loss'] for x in rr]); t=np.array([x['tail_q90_q50'] for x in rr])
            agg.append({'drop':drop,'method':method,'p':args.power if method=='hifedprox' else (2 if method=='fedprox' else None),'test_acc_mean':float(a.mean()),'test_acc_sd':float(a.std(ddof=1)) if len(a)>1 else 0.0,'test_loss_mean':float(l.mean()),'test_loss_sd':float(l.std(ddof=1)) if len(l)>1 else 0.0,'tail_mean':float(t.mean()),'tail_sd':float(t.std(ddof=1)) if len(t)>1 else 0.0})
    with open(out/'aggregate.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
    (out/'metadata.json').write_text(json.dumps({'args':vars(args),'R':R,'dataset_stats':stats},indent=2)); print('AGGREGATE',json.dumps(agg,indent=2),flush=True)
if __name__=='__main__': main()
