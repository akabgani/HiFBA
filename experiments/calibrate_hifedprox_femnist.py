#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np
import hifedprox_fedprox_femnist as core

def split_train_validation(users, data, fraction, seed):
    fit, val = {}, {}
    for j,u in enumerate(users):
        X,y=data[u]; n=len(y)
        if n < 2:
            fit[u]=(X.copy(),y.copy()); val[u]=(X[:0].copy(),y[:0].copy()); continue
        rs=np.random.RandomState(seed+j)
        perm=rs.permutation(n)
        nv=max(1,int(round(fraction*n))); nv=min(nv,n-1)
        vidx=perm[:nv]; tidx=perm[nv:]
        fit[u]=(X[tidx].copy(),y[tidx].copy()); val[u]=(X[vidx].copy(),y[vidx].copy())
    return fit,val

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',required=True); ap.add_argument('--out',default='calibration')
    ap.add_argument('--power',type=float,required=True); ap.add_argument('--r-scale',type=float,required=True)
    ap.add_argument('--validation-fraction',type=float,default=0.15); ap.add_argument('--validation-seed',type=int,default=20261027)
    ap.add_argument('--rounds',type=int,default=200); ap.add_argument('--clients-per-round',type=int,default=10); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=10)
    ap.add_argument('--lr',type=float,default=0.003); ap.add_argument('--mu',type=float,default=1.0); ap.add_argument('--num-classes',type=int,default=26); ap.add_argument('--drop',type=float,default=0.9)
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); ap.add_argument('--pilot-rounds',type=int,default=20); ap.add_argument('--eval-every',type=int,default=10)
    args=ap.parse_args()
    users,full_train=core.merge_jsons(args.train)
    fit,val=split_train_validation(users,full_train,args.validation_fraction,args.validation_seed)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    nfit=sum(len(fit[u][1]) for u in users); nval=sum(len(val[u][1]) for u in users)
    pilot=core.run_method(users,fit,val,'fedprox',0.0,args.pilot_rounds,args.clients_per_round,args.epochs,args.batch_size,args.lr,args.mu,2.0,1.0,0,args.num_classes,max(args.pilot_rounds,1))
    base_R=max(float(pilot['update_norm_q90_all']),1e-8); R=base_R*args.r_scale
    rows=[]; histories=[]
    for seed in args.seeds:
        res=core.run_method(users,fit,val,'hifedprox',args.drop,args.rounds,args.clients_per_round,args.epochs,args.batch_size,args.lr,args.mu,args.power,R,seed,args.num_classes,args.eval_every)
        f=res['final']; rows.append({'power':args.power,'r_scale':args.r_scale,'base_R':base_R,'R':R,'seed':seed,'val_accuracy':f['test_acc'],'val_loss':f['test_loss'],'fit_accuracy':f['train_acc'],'fit_loss':f['train_loss'],'tail_q90_q50':f['tail_q90_q50']}); histories.append(res)
    a=np.array([r['val_accuracy'] for r in rows]); l=np.array([r['val_loss'] for r in rows]); t=np.array([r['tail_q90_q50'] for r in rows])
    aggregate={'power':args.power,'r_scale':args.r_scale,'base_R':base_R,'R':R,'drop':args.drop,'n_fit':nfit,'n_validation':nval,'val_accuracy_mean':float(a.mean()),'val_accuracy_sd':float(a.std(ddof=1)) if len(a)>1 else 0.0,'val_loss_mean':float(l.mean()),'val_loss_sd':float(l.std(ddof=1)) if len(l)>1 else 0.0,'tail_mean':float(t.mean()),'tail_sd':float(t.std(ddof=1)) if len(t)>1 else 0.0}
    with open(out/'summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(out/'aggregate.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate.keys())); w.writeheader(); w.writerow(aggregate)
    (out/'metadata.json').write_text(json.dumps({'args':vars(args),'aggregate':aggregate},indent=2))
    (out/'histories.json').write_text(json.dumps(histories))
    print('CALIBRATION_AGGREGATE',json.dumps(aggregate,indent=2),flush=True)
if __name__=='__main__': main()
