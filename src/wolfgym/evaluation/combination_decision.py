"""Public-only RR/RL/LR/LL decision adapter; no provider or credentials.

Create a fresh adapter for every episode. The trusted caller supplies Gym info
and an optional request function taking only a public request dictionary.
Request persistence, transport retries and spending limits belong to the host.
"""
import copy
import json

from wolfgym.evaluation.combination_public import (
    BaselinePolicy, PublicHistory, canonical_json, digest, public_frame, public_target,
)


DECISION_VERSION = 'combination-decision-v1'
SYSTEM = (
    'You regulate a simulated financial market. Use the supplied public evidence '
    'to protect ordinary participants from wealth losses while limiting harm to '
    'normal market activity. There are 30 decision days, numbered 0 through 29. '
    'WAIT costs zero. FREEZE blocks trading in one asset for the next single '
    'market day, costs 4, and then expires. The initial administrative budget '
    'is 12. Only the current remaining budget and legal actions apply. '
    'Asset and entity identifiers are opaque references. Observations are '
    'ordered from oldest to newest; interventions list your executed freezes. '
    'Return exactly the requested JSON object, without commentary or markdown.'
)


def response_schema(mode, targets):
    """Schema is also supplied to a future transport for structured output."""
    target = {'type': 'string', 'enum': list(targets)}
    if mode == 'RL':
        variants = [({'target_id': target}, ['target_id'])]
    elif mode == 'LR':
        variants = [({'action': {'type': 'string', 'enum': ['WAIT', 'FREEZE']}}, ['action'])]
    elif mode == 'LL':
        variants = [({'action': {'const': 'WAIT'}}, ['action']),
                    ({'action': {'const': 'FREEZE'}, 'target_id': target}, ['action', 'target_id'])]
    else:
        raise ValueError('only model conditions have a response schema')
    objects = [{'type': 'object', 'properties': p, 'required': r, 'additionalProperties': False}
               for p, r in variants]
    return objects[0] if len(objects) == 1 else {'oneOf': objects}


def build_request(view, mode):
    responsibility = {
        'RL': 'The public timing rule has scheduled a FREEZE now. Choose one legal asset: {"target_id":"..."}.',
        'LR': 'Choose whether to WAIT or FREEZE now: {"action":"WAIT"} or {"action":"FREEZE"}. A public rule selects the asset after your decision.',
        'LL': 'Choose {"action":"WAIT"} or {"action":"FREEZE","target_id":"..."}.',
    }
    if mode not in responsibility:
        raise ValueError('unknown model condition')
    return {'version': DECISION_VERSION, 'system': SYSTEM + ' ' + responsibility[mode],
            'user': canonical_json(view),
            'response_schema': response_schema(mode, view['observations'][-1]['action_targets'])}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def parse_response(raw, mode, frame):
    """Malformed, extra-field, unknown-target and unauthorized answers fail."""
    if type(raw) is not str:
        raise ValueError('response must be JSON text')
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if type(value) is not dict:
        raise ValueError('response must be one JSON object')
    if mode == 'RL':
        if set(value) != {'target_id'}:
            raise ValueError('target condition requires only target_id')
        target = value['target_id']
    elif mode in ('LR', 'LL'):
        action = value.get('action')
        if action == 'WAIT' and set(value) == {'action'}:
            return 0
        expected = {'action'} if mode == 'LR' else {'action', 'target_id'}
        if action != 'FREEZE' or set(value) != expected:
            raise ValueError('invalid action response')
        target = public_target(frame) if mode == 'LR' else value['target_id']
    else:
        raise ValueError('unknown model condition')
    if type(target) is not str or target not in frame['action_targets']:
        raise ValueError('unknown public asset')
    action = frame['action_targets'].index(target) + 1
    if not frame['action_mask'][action]:
        raise ValueError('unaffordable action')
    return action


class CombinationDecisionAdapter:
    """Each call consumes one successive decision frame on its own trajectory."""
    def __init__(self, mode, detector, request=None, *, window=5):
        if mode not in ('RR', 'RL', 'LR', 'LL'):
            raise ValueError('unknown responsibility condition')
        if mode != 'RR' and not callable(request):
            raise ValueError('model conditions require a request function')
        self.mode = mode
        self.request = request
        self.history = PublicHistory(window)
        self.rule = BaselinePolicy({'id': 'alarm_refresh', 'family': 'alarm_refresh'}, detector)
        self.log = []

    def __call__(self, info):
        frame = public_frame(info)
        if frame['day'] != len(self.log) or not 0 <= frame['day'] < 30:
            raise ValueError('fresh episode adapter must consume days 0..29 in order')
        # Build tentative state so transport failures do not consume a decision.
        history = copy.deepcopy(self.history)
        rule = copy.deepcopy(self.rule)
        view = history.observe(frame)
        action = 0
        payload = response = error = None
        if frame['remaining_budget'] >= 4:
            scheduled = rule(view) if self.mode in ('RR', 'RL') else None
            if self.mode == 'RR':
                action = scheduled
            elif self.mode != 'RL' or scheduled:
                payload = build_request(view, self.mode)
                # Infrastructure failures propagate; they are not model WAITs.
                response = self.request(copy.deepcopy(payload))
                try:
                    action = parse_response(response, self.mode, frame)
                except (ValueError, TypeError, RecursionError) as exc:
                    error = type(exc).__name__ + ': ' + str(exc)
                    action = 0
        history.record(frame, action)
        self.log.append({'day': frame['day'], 'frame': frame, 'input_sha256': digest(view),
                         'request': payload, 'response': response, 'invalid_action': error is not None,
                         'error': error, 'action': action, 'cost': 4. * bool(action)})
        self.history, self.rule = history, rule
        return action
