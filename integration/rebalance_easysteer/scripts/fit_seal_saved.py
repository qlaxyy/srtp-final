"""Pool fixed SEAL classes from saved-answer replay; no search or generation."""
import argparse
import json
from pathlib import Path
import numpy as np
from mechanism_candidates import ROOT,BASE,read,save,sha,require,write_vector,cosine


def pooled_direction(sums,counts,indices):
    total=sums[indices].sum(0);n=counts[indices].sum(0)
    require(np.all(n>0),'Missing behavioral class')
    return total[0]/n[0]-(total[1]+total[2])/(n[1]+n[2])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();require(not out.exists(),'Immutable output exists')
    ledger=read(a.replay/'ledger.json');plan=read(a.replay/'plan.json')
    require(ledger['status']=='completed' and ledger['new_answers']==0 and ledger['all_reference_states_exact'],'Incomplete or mismatched replay')
    for name,digest in ledger['files_sha256'].items():require(sha(a.replay/name)==digest,'Replay file changed')
    sums=np.load(a.replay/'class_sums.npy');counts=np.load(a.replay/'class_counts.npy')
    require(sums.shape==(500,3,1536) and counts.shape==(500,3),'Wrong replay shape')
    require(np.isfinite(sums).all() and (counts>=0).all(),'Invalid sums/counts')
    require(counts.sum(0).tolist()==[55713,23538,4416],'Class counts changed')
    backup=ROOT/read(ROOT/BASE/'configs/overnight_research_20260912.json')['first_investigation']['inputs']['backup']
    selection=read(backup/'selected_layer.json');train=selection['training_questions'];valid=selection['validation_questions']
    require(len(train)==400 and len(valid)==100 and not set(train)&set(valid),'Wrong original development groups')
    vector=pooled_direction(sums,counts,np.arange(500)).astype(np.float32)
    d_train=pooled_direction(sums,counts,train);d_valid=pooled_direction(sums,counts,valid)
    require(np.isfinite(vector).all() and np.linalg.norm(vector)>0,'Invalid vector')
    out.mkdir(parents=True);write_vector(out/'seal_vector.pt',vector,backup/'auto_vector.pt')
    fit=dict(status='fixed_author_formula_self_calibrated_asset',model=plan['model'],decoder_output_layer=19,source_hidden_state_index=20,
        algorithm='seal',static_coefficient=1.,normalize=False,vector_sha256=sha(out/'seal_vector.pt'),
        formula='pooled execution mean minus pooled union(reflection,transition) mean; all500 saved responses; raw vector, alpha1',
        raw_vector_norm=float(np.linalg.norm(vector.astype(float))),class_steps=counts.sum(0).tolist(),class_questions=(counts>0).sum(0).tolist(),
        original_development_split_direction_cosine=cosine(d_train,d_valid),new_answers=0,new_model_forwards=0,
        original_author_source=plan['original_source'],replay_ledger_sha256=sha(a.replay/'ledger.json'),replay_files_sha256=ledger['files_sha256'],
        generations_sha256=plan['generations_sha256'],model_files_sha256=plan['model_files_sha256'],
        limitations=['Self-calibrated adaptation using existing500 greedy answers, not author1000 correct/incorrect selection.',
            'Original400/100 split was used for layer selection; stability is development evidence only.',
            'Keep project prompt, sample0.7/0.95/42 and cap16000 for comparison; differs from paper protocol.',
            'Direction is applied on generated delimiter inputs after decoder19, within think markers only.'],
        source_sha256=sha(Path(__file__),source=True))
    save(out/'fit.json',fit);print(json.dumps({k:v for k,v in fit.items() if k not in ['original_author_source','model_files_sha256']}))


if __name__=='__main__':main()
