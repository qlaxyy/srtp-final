"""Reuse validated synchronous replay lifecycle for the large lexical state."""
from label_alignment_adapter import AlignmentAdapter
from replay_adapter import ReplayAdapter

class ReplayAlignmentAdapter(AlignmentAdapter,ReplayAdapter):
    fields=ReplayAdapter.fields+('lex_state','lex_hit','lex_ready','closed_hit','lex_open','lex_changes')

    def add_requests(self,output):
        fresh=[r.req_id for r in output.scheduled_new_reqs if r.req_id not in self.suspended]
        ReplayAdapter.add_requests(self,output)
        for rid in fresh:
            i=self.active[rid]
            self.lex_state[i]=self.lex_changes[i]=0
            self.lex_hit[i]=self.lex_ready[i]=self.closed_hit[i]=self.lex_open[i]=False

    def remove_request(self,rid):
        suspended=rid in self.suspending
        changes=int(self.lex_changes[self.active[rid]].cpu()) if rid in self.active and not suspended else None
        result=ReplayAdapter.remove_request(self,rid)
        if changes is not None:self.completed[rid]['lexical_control_changes']=changes
        return result
