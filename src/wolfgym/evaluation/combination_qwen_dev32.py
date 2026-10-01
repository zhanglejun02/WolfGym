"""Fresh development batch using the previously tested bounded Qwen request path."""
import time
from wolfgym.evaluation.combination_host import read_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_qwen_pilot import RESERVATION,QwenTransport
from wolfgym.evaluation.combination_qwen_recovery import RecoveryJournal
from wolfgym.evaluation.combination_qwen_incomplete_recovery import IncompleteRecoveryJournal
from wolfgym.evaluation.combination_requests import RequestJournal

VERSION='combination-qwen-dev32-v1'


class DevelopmentJournal(RecoveryJournal):
    # Reuse the tested network/empty receipt classifier and bounded request loop.
    # This is a fresh dataset, not a reset of the exhausted calibration requests.
    request=IncompleteRecoveryJournal.request
    stop=IncompleteRecoveryJournal.stop

    def __init__(self,output,manifest,job,*,pause=time.sleep,transport_factory=QwenTransport):
        from pathlib import Path
        self.output,self.manifest,self.job=Path(output),manifest,job
        self.pause,self.transport_factory=pause,transport_factory
        self.amendment={'original_ledgers':{}}
        RequestJournal.__init__(self,self.output/(job['id']+'.sqlite'),
            contract={'manifest_sha256':digest(manifest),'job':job['id'],'version':VERSION,'limits_scope':'all_jobs_pooled'},
            max_attempts=manifest['max_requests_total'],max_cost_microusd=manifest['max_cost_microusd_total'],
            reservation_microusd=RESERVATION,retries=2)

    def verify_incident(self):
        # The shared request loop calls this immediately before reserving a POST.
        if read_json(self.output/'manifest.json')!=self.manifest:
            raise ValueError('fresh batch manifest drift')
