import json,sys
for f in sys.argv[1:]:
    d=json.load(open(f))
    for s,r in d.get("seeds",{}).items():
        print(f"{f.split('exp_')[-1][:-5]} seed{s} best_ep={r.get('best_epoch')} ran={r.get('epochs_run')} stop={r.get('stopped')} rho={r.get('spearman_norm_vs_count')} sec={round(r.get('train_seconds',0))} mem={r.get('memory_overhead')}")
        print("  val_curve", {k: round(100*v,2) for k,v in r.get('val_curve',{}).items()})
        for k,e in r.get("eval",{}).items():
            print(f"  {k:12s} beta={e['best_beta']:<5} val R={100*e['val']['recall']:.2f} N={100*e['val']['ndcg']:.2f} | test R={100*e['test']['recall']:.2f} N={100*e['test']['ndcg']:.2f}")
