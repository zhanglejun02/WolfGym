"""Input-only calibration composed with the frozen four-condition adapter."""
from pathlib import Path

from wolfgym.evaluation.combination_decision import CombinationDecisionAdapter, build_request
from wolfgym.evaluation.combination_qwen_pilot import pilot_request, VERSION as TRANSPORT_VERSION

VERSION = 'combination-qwen-calibration-v2'
DICTIONARY_PATH = Path(__file__).with_name('combination_input_dictionary_v2.txt')
DICTIONARY = DICTIONARY_PATH.read_text(encoding='utf-8').strip()


def calibrated_request(view, mode):
    # The same corrected permissions as the accepted pilot; only definitions
    # are appended for LL. Transport metadata is not sent as model content.
    payload = pilot_request(view)
    if mode != 'LL':
        original = build_request(view, 'LL')
        alternate = build_request(view, mode)
        suffix = original['system'].rsplit('Return exactly', 1)[1]
        replacement = alternate['system'].rsplit('Return exactly', 1)[1]
        payload['system'] = payload['system'].replace('Return exactly'+suffix, 'Return exactly'+replacement)
        payload['response_schema'] = alternate['response_schema']
    payload['system'] += '\n\n' + DICTIONARY
    payload['version'] = TRANSPORT_VERSION
    payload['calibration_version'] = VERSION
    payload['mode'] = mode
    return payload


class CalibratedAdapter:
    """Reuse policy, parsing, history, error and transactional-state semantics."""
    def __init__(self, mode, detector, request=None):
        import json
        def calibrated_call(original):
            return request(calibrated_request(json.loads(original['user']), mode))
        self.inner = CombinationDecisionAdapter(mode, detector, calibrated_call if mode != 'RR' else None)

    @property
    def log(self):
        return self.inner.log

    def __call__(self, info):
        import json
        action = self.inner(info)
        row = self.log[-1]
        if row['request'] is not None:
            row['request'] = calibrated_request(json.loads(row['request']['user']), self.inner.mode)
        return action
