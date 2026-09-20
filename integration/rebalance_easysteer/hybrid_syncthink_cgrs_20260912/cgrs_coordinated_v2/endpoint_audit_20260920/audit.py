"""CPU endpoint composition and geometry audit; no model or label changes."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).resolve().parent; C=H.parent
B=next(p for p in H.parents if (p/'.git').exists())
OLD=Path('E:/srtp/srtp-final/.codex_work/question_balanced_20260911/original_selected_layer')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cos(a,b):return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output;out.mkdir(exist_ok=False)
    archive=B/'.codex_work/online_endpoint_full_20260919/evidence.tar.gz'
    assert sha(archive)=='210462946624e2447ab13e7db9e9f9785b67edd4812a62403f04c02cbeeb2233'
    native_labels=None;records={}
    with tarfile.open(archive,'r|gz') as t:
        for m in t:
            parts=m.name.split('/')
            if parts[0] not in ['online_endpoint_full_20260919_collect_run1','online_endpoint_full_20260919_recovery52_run1']:continue
            if parts[-1]=='L27_labels.json':native_labels=json.load(t.extractfile(m))
            if len(parts)==3 and parts[1].startswith('L27_batch') and parts[-1]=='result.json':
                d=json.load(t.extractfile(m));offset=int(parts[1].split('batch')[1])
                for j,r in enumerate(d['records']):
                    assert offset+j not in records;records[offset+j]=r
    assert len(native_labels)==len(records)==500
    rows=read(C/'online_endpoint_full_20260919/rows.json')
    oldlabels=[r['U'] for r in read(C/'self_feedback_train500_20260918_run1/labels.json')]
    for i in range(500):
        assert oldlabels[i]['problem_sha256']==native_labels[i]['problem_sha256']==records[i]['problem_sha256']==rows[i]['problem_sha256']
    sources={
      'old':(OLD/'steps.json',OLD/'layer_21.npy',oldlabels,OLD/'fit.json'),
      'native_q90':(B/'.codex_work/under_q90_20260920/run1/L27/q90/steps.json',B/'.codex_work/under_q90_20260920/run1/L27/layer_21.npy',native_labels,C/'under_q90_20260920/assets/L27/q90/fit.json')}
    report={'scope':'Training descriptive diagnostic; cap/wrong are proxies, not step labels; native collection differs in execution path.', 'new_answers':0,'model_forwards':0,'groups':{},'inputs':{}}
    means={};sample_indices={};stepdata={}
    for name,(sp,xp,lab,fp) in sources.items():
        s=read(sp);stepdata[name]=s;x=np.load(xp,mmap_mode='r');assert len(s)==len(x)
        f=read(fp);p=f['parameters'];conf=np.array([a['confidence'] for a in s]);lex=np.array([a['lexical_hit'] for a in s]);q=np.array([a['question'] for a in s])
        masks={'O':lex|(conf<p['q25c']),'U':(~lex)&(conf>p['q75c'])}
        cap=np.array([a['tokens']==16000 for a in lab]);wrong=np.array([not a['correct'] for a in lab])
        result={'trajectory_count':500,'capped_trajectories':int(cap.sum()),'wrong_trajectories':int(wrong.sum()),'steps':len(s),'endpoints':{}}
        means[name]={}
        for side,mask in masks.items():
            counts=np.bincount(q[mask],minlength=500);indices=np.flatnonzero(mask)
            mu=x[mask].mean(0,dtype=np.float64);means[name][side]=mu
            top=np.argsort(-counts,kind='stable')[:10];nocap=mask&~cap[q]
            shift=x[nocap].mean(0,dtype=np.float64)-mu
            perq=np.stack([x[mask&(q==i)].mean(0,dtype=np.float64) for i in np.flatnonzero(counts)])
            leave=[np.linalg.norm((counts.sum()*mu-counts[i]*x[mask&(q==i)].mean(0,dtype=np.float64))/(counts.sum()-counts[i])-mu) for i in top]
            result['endpoints'][side]={'steps':int(mask.sum()),'questions':int((counts>0).sum()),'from_capped':int(cap[q[mask]].sum()),'from_wrong':int(wrong[q[mask]].sum()),
              'top10_steps':int(counts[top].sum()),'top10_questions':top.tolist(),'top10_counts':counts[top].tolist(),
              'effective_questions':float(counts.sum()**2/(counts@counts)), 'mean_norm':float(np.linalg.norm(mu)),
              'exclude_caps_mean_shift_norm':float(np.linalg.norm(shift)),
              'question_equal_mean_shift_norm':float(np.linalg.norm(perq.mean(0)-mu)),
              'max_single_top_question_removal_shift_norm':float(max(leave)),
              'paired_to_original_shorter_steps':int(sum(lab[i]['tokens']<oldlabels[i]['tokens'] for i in q[mask]))}
            # Fixed descriptive sample: two largest contributors at median selected step, four random steps.
            rng=np.random.default_rng(20260920+(side=='U'))
            chosen=[int(np.flatnonzero(mask&(q==i))[counts[i]//2]) for i in top[:2]]+rng.choice(indices,4,replace=False).tolist()
            sample_indices[name+'_'+side]=list(dict.fromkeys(chosen))
        v=means[name]['O']-means[name]['U'];saved=torch.load(fp.parent/'auto_vector.pt',map_location='cpu',weights_only=True).numpy()
        assert np.array_equal(v.astype(np.float32),saved)
        result['vector_norm']=float(np.linalg.norm(v));report['groups'][name]=result
        report['inputs'][name]={str(p):sha(p) for p in [sp,xp,fp,fp.parent/'auto_vector.pt']}
    vo=means['old']['O']-means['old']['U'];vn=means['native_q90']['O']-means['native_q90']['U']
    do=means['native_q90']['O']-means['old']['O'];du=means['native_q90']['U']-means['old']['U']
    report['geometry']={'O_shift_norm':float(np.linalg.norm(do)),'U_shift_norm':float(np.linalg.norm(du)),
      'shift_cosine':cos(do,du),'vector_change_norm':float(np.linalg.norm(vn-vo)),'old_new_vector_cosine':cos(vo,vn),
      'only_O_update':{'norm':float(np.linalg.norm(vo+do)),'cosine_to_old':cos(vo,vo+do)},
      'only_U_update':{'norm':float(np.linalg.norm(vo-du)),'cosine_to_old':cos(vo,vo-du)}}
    np.savez(out/'endpoint_means.npz',old_O=means['old']['O'],old_U=means['old']['U'],new_O=means['native_q90']['O'],new_U=means['native_q90']['U'])
    tokpath=B/'.codex_work/wsc_weights_20260917/tokenizer.json';td=read(tokpath)
    assert td['decoder']['type']=='ByteLevel' and td['model']['type']=='BPE'
    vocab={v:k for k,v in td['model']['vocab'].items()}
    vocab.update({a['id']:a['content'] for a in td['added_tokens']})
    bytevalues=list(range(33,127))+list(range(161,173))+list(range(174,256));chars=bytevalues.copy();extra=0
    for b in range(256):
        if b not in bytevalues:bytevalues.append(b);chars.append(256+extra);extra+=1
    reverse={chr(c):b for c,b in zip(chars,bytevalues)}
    def decode(ids):return bytes(reverse[c] for i in ids for c in vocab[i]).decode('utf-8',errors='replace')
    samples=[]
    for side in ['O','U']:
        for index in sample_indices['native_q90_'+side]:
            s=stepdata['native_q90'][index];i=s['question'];ids=records[i]['token_ids'];a=s['start'];b=s['stop']
            samples.append({'side':side,'step_index':index,'step':s,'label':native_labels[i],'train_index':rows[i]['train_index'],
              'problem':rows[i]['problem'],'context':decode(ids[max(0,a-180):min(len(ids),b+180)]),
              'step_text':decode(ids[a:b])})
    save(out/'samples.json',samples);save(out/'registry.json',[dict(question=i,train_index=r['train_index'],problem_sha256=r['problem_sha256'],use='existing calibration descriptive audit') for i,r in enumerate(rows)])
    report['archive_sha256']=sha(archive);report['tokenizer_sha256']=sha(tokpath)
    save(out/'summary.json',report);print(json.dumps(report['groups'],indent=2));print(json.dumps(report['geometry'],indent=2))
if __name__=='__main__':main()
