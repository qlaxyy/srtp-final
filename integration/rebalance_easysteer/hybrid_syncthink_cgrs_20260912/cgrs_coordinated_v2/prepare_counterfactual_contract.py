"""Record implemented contract and outstanding engine integration honestly."""
from prepare_length_vector import HERE,read,save,sha
from test_counterfactual_snapshot import main as check

def main():
    check();out=HERE/'counterfactual_contract_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
    parent=HERE/'length_sign_ablation_20260918_run1/release.json';release=read(parent)
    save(out/'receipt.json',dict(status='worker_snapshot_contract_implemented_CPU_checked_not_GPU_ready',
        source_sha256={n:sha(HERE/n) for n in ('counterfactual_snapshot.py','test_counterfactual_snapshot.py','prepare_counterfactual_contract.py')},
        cpu_checks=10,model='DeepSeek-R1-Distill-Qwen-1.5B',
        implemented=['Capture all seven native typed fields, actual injection history and thirteen lexical fields without changing parent.',
        'Fork independent copies; skip changes only the last unconsumed historical injection. Preserve coefficient and lexical state.',
        'Stage through native and lexical suspension maps only after validating assets, exact prompt/generated token identity, clocks and request uniqueness.',
        'Reject async and ordinary long-prompt substitution. Explicit cleanup of staged, not active requests.'],
        not_implemented=['Scheduler admission of generated-prefix sibling requests.',
        'Live boundary capture after accepted-token update but before consuming the boundary.',
        'GPU replay/no-op equivalence, model KV and probability equality, exact target-only displacement audit.'],
        intended_engineering=dict(parent_release_sha256=sha(parent),rows=release['engineering_rows'],
            purpose='Previously exposed eight training engineering questions only; no independent accuracy or compression estimate.',
            stage1='L27 reference versus snapshot-only observer, each <=512 tokens, 16 short outputs. No-op outputs/history must be exactly equal; at least one complete boundary per request required.',
            stage2='Only after stage1: reconstruct saved siblings with same partition and verify two no-op continuations, then target-only skip. Exact budget fixed with runnable engine integration, not authorized to launch from this receipt.',
            execution='Synchronous vLLM native replay; no environment changes or shared controller edits.',
            stop=['No applicable boundary','Parent mutation','Prefix/history mismatch','Output divergence in no-op arms','Wrong target mask','Preemption outside verified lifecycle','OOM or nonfinite state']),
        vector_research='Outcome labels condition existing over/under extraction pools; do not use action-preference means as a new direction by default. Original-vector choice policy is a separate research result. New directions require held-out generation validation.',
        gpu_authorization='User authorization persists; readiness is missing engine implementation, not permission.',
        new_answers=0,new_forward=0))

if __name__=='__main__':main()
