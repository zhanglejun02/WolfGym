"""Bounded Qwen pilot transport and accurate LL prompt; no research/test entry."""
import copy
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import urllib.request

from wolfgym.evaluation.combination_decision import build_request, parse_response
from wolfgym.evaluation.combination_public import PublicHistory, public_frame, digest, canonical_json
from wolfgym.evaluation.combination_requests import Reply, TransportFailure
from wolfgym.evaluation.combination_model_host import durable_json

MODEL = 'qwen/qwen3.8-max-0902'
CANONICAL = 'qwen/qwen3.8-max-20260902'
VERSION = 'combination-qwen-pilot-v1'
# text-only UTF-8 BPE byte bound plus conservative chat-framing allowance.
MAX_BODY_BYTES = 24000
INPUT_BOUND = MAX_BODY_BYTES + 4096
MAX_OUTPUT = 4096
RESERVATION = 110000  # USD micro: input <= 28096 * $3/M + output <=4096 * $6/M.
SETTINGS = {'model': MODEL, 'temperature': 0, 'seed': 20260928, 'max_tokens': MAX_OUTPUT,
            'reasoning': {'effort': 'low', 'exclude': True}, 'stream': False,
            'response_format': {'type': 'json_object'},
            'provider': {'only': ['alibaba'], 'allow_fallbacks': False, 'require_parameters': True,
                         'max_price': {'prompt': 3, 'completion': 6, 'request': 0}}}


def pilot_request(view):
    p = build_request(view, 'LL')
    old = 'FREEZE blocks trading in one asset for the next single market day, costs 4, and then expires.'
    new = ('FREEZE applies full-strength controls to the chosen asset for the next single market day: '
           'it suppresses associated messages and trade, spoof, and wash orders, costs 4, and then expires. '
           'This is a bundled simulation control, not just a trading halt.')
    if old not in p['system']:
        raise ValueError('parent prompt changed')
    p['system'] = p['system'].replace(old, new); p['version'] = VERSION
    return p


class PilotAdapter:
    def __init__(self, request):
        self.request = request; self.history = PublicHistory(5); self.log = []

    def __call__(self, info):
        frame = public_frame(info)
        if frame['day'] != len(self.log) or not 0 <= frame['day'] < 30:
            raise ValueError('episode/day isolation mismatch')
        history = copy.deepcopy(self.history); view = history.observe(frame)
        payload = response = error = None; action = 0
        if frame['remaining_budget'] >= 4:
            payload = pilot_request(view); response = self.request(copy.deepcopy(payload))
            try: action = parse_response(response, 'LL', frame)
            except (ValueError, TypeError, RecursionError) as exc:
                error = type(exc).__name__ + ': ' + str(exc)
        history.record(frame, action)
        self.log.append({'day': frame['day'], 'frame': frame, 'input_sha256': digest(view),
                         'request': payload, 'response': response, 'invalid_action': error is not None,
                         'error': error, 'action': action, 'cost': 4.*bool(action)})
        self.history = history
        return action


def wire_body(payload):
    if payload['version'] != VERSION:
        raise ValueError('incorrect prompt version')
    body = {**copy.deepcopy(SETTINGS), 'messages': [
        {'role': 'system', 'content': payload['system']}, {'role': 'user', 'content': payload['user']}]}
    encoded = canonical_json(body).encode()
    if len(encoded) > MAX_BODY_BYTES:
        raise ValueError('request exceeds predeclared input bound')
    return encoded


def decode_reply(data):
    if data.get('model') not in (MODEL, CANONICAL) or data.get('provider') != 'Alibaba':
        raise ValueError('provider/model identity drift')
    u = data['usage']
    for k in ('prompt_tokens', 'completion_tokens'):
        if type(u[k]) is not int or u[k] < 0:
            raise ValueError('invalid usage')
    if u['prompt_tokens'] > INPUT_BOUND or u['completion_tokens'] > MAX_OUTPUT:
        raise ValueError('provider token bound violated')
    cost = Decimal(str(u['cost']))
    if not cost.is_finite() or cost < 0:
        raise ValueError('missing/invalid provider cost')
    charge = int((cost*1000000).to_integral_value(rounding=ROUND_CEILING))
    if charge > RESERVATION:
        raise ValueError('provider cost bound violated')
    message = data['choices'][0]['message']
    content = message.get('content')
    if content is None: content = ''  # length/refusal -> invalid model output, recorded WAIT.
    if not isinstance(content, str): raise ValueError('unexpected content type')
    usage = {'input_tokens': u['prompt_tokens'], 'output_tokens': u['completion_tokens'],
             'reasoning_tokens': u.get('completion_tokens_details', {}).get('reasoning_tokens', 0),
             'cached_input_tokens': u.get('prompt_tokens_details', {}).get('cached_tokens', 0)}
    usage = {k: v for k, v in usage.items() if v is not None}
    return Reply(content, charge, usage)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise TransportFailure('redirects disabled')


class QwenTransport:
    """One explicit HTTP POST, no retry/fallback, no credential in artifacts."""
    def __init__(self, key, directory):
        self.key = key; self.directory = Path(directory)
        self.opener = urllib.request.build_opener(NoRedirect())

    def __call__(self, payload):
        encoded = wire_body(payload); identity = digest(payload)
        dest = self.directory/f'{identity}.json'
        if dest.exists():
            raise TransportFailure('unreconciled provider response; do not repeat POST')
        req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', data=encoded,
              headers={'Authorization': 'Bearer '+self.key, 'Content-Type': 'application/json',
                       'X-Title': 'WolfGym bounded train pilot'}, method='POST')
        try:
            with self.opener.open(req, timeout=120) as response:
                data = json.load(response)
        except Exception as exc:
            # Neither credentials nor server exception bodies go into logs.
            durable_json(self.directory/f'{identity}.failure.json', {'exception_type': type(exc).__name__,
                         'http_status': getattr(exc, 'code', None), 'payload_sha256': identity})
            raise TransportFailure('single provider attempt failed; reservation retained') from None
        # Record a bounded allowlist, never arbitrary provider headers or reasoning text.
        message = data.get('choices', [{}])[0].get('message', {})
        safe = {'id': data.get('id'), 'model': data.get('model'), 'provider': data.get('provider'),
                'usage': data.get('usage'), 'created': data.get('created'),
                'choices': [{'finish_reason': data.get('choices', [{}])[0].get('finish_reason'),
                             'message': {'content': message.get('content'), 'refusal': message.get('refusal')}}]}
        durable_json(dest, safe)
        return decode_reply(safe)
