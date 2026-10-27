#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np
import hifedprox_fedprox_femnist as core

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',required=True); ap.add_argument('--test',required=True); ap.add_argument('--out',default='final_results')
    ap.add_argument('--method',choices=['fedavg','fedprox','hifedprox'],required=True); ap.add_argument('--drop',type=float,required=True)
    ap.add_argument('--rounds',type=int,default=200); ap.add_argument('--clients-per-round',type=int,default=10); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=10)
    ap.add_argument('--lr',type=float,default=0.003); ap.add_argument('--num-classes',type=int,default=26); ap.add_argument('--seeds',type=int,nargs='+',default=[8,9,10,11,12]); ap.add_argument('--eval-every',type=int,default=5)
    args=ap.parse_args()
    users,tr=core.merge_jsons(args.train); _,te=core.merge_jsons(args.test); users=[u for u in users if u in te]
    stats=core.dataset_stats(users,tr,te)
    # Frozen training-only calibration procedure for HiFedProx:
    # p=3, mu=2, R=0.5 * q90 client-update norm from a 20-round quadratic
    # pilot with the same mu=2. FedProx remains the paper/released-code mu=1 baseline.
    if args.method=='hifedprox':
        pilot=core.run_method(users,tr,te,'fedprox',0.0,20,args.clients_per_round,args.epochs,args.batch_size,args.lr,2.0,2.0,1.0,0,args.num_classes,20)
        base_R=max(float(pilot['update_norm_q90_all']),1e-8); R=0.5*base_R; mu=2.0; p=3.0
    elif args.method=='fedprox':
        base_R=None; R=1.0; mu=1.0; p=2.0
    else:
        base_R=None; R=1.0; mu=0.0; p=2.0
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; histories=[]
    for seed in args.seeds:
        res=core.run_method(users,tr,te,args.method,args.drop,args.rounds,args.clients_per_round,args.epochs,args.batch_size,args.lr,mu,p,R,seed,args.num_classes,args.eval_every)
        f=res['final']; histories.append(res)
        rows.append({'method':args.method,'drop':args.drop,'seed':seed,'p':p if args.method!='fedavg' else '', 'mu':mu,'R':R,'base_R':base_R if base_R is not None else '', 'test_accuracy':f['test_acc'],'test_loss':f['test_loss'],'train_accuracy':f['train_acc'],'train_loss':f['train_loss'],'tail_q90_q50':f['tail_q90_q50'],'all_update_q90':res['update_norm_q90_all'],'all_update_q50':res['update_norm_q50_all']})
    with (out/'summary.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    def ms(key):
        a=np.array([float(r[key]) for r in rows]); return float(a.mean()),float(a.std(ddof=1))
    acc_m,acc_s=ms('test_accuracy'); loss_m,loss_s=ms('test_loss'); tail_m,tail_s=ms('tail_q90_q50')
    agg={'method':args.method,'drop':args.drop,'p':p if args.method!='fedavg' else None,'mu':mu,'R':R,'base_R':base_R,'n_seeds':len(rows),'test_accuracy_mean':acc_m,'test_accuracy_sd':acc_s,'test_loss_mean':loss_m,'test_loss_sd':loss_s,'tail_mean':tail_m,'tail_sd':tail_s,'dataset_stats':stats}
    (out/'aggregate.json').write_text(json.dumps(agg,indent=2)); (out/'histories.json').write_text(json.dumps(histories))
    print('FINAL_AGGREGATE',json.dumps(agg,indent=2),flush=True)
if __name__=='__main__': main()
