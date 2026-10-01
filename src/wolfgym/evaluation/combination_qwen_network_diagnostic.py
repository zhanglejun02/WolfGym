"""Same Qwen wire request; retain a credential-redacted causal error chain."""
import json
import re
import time
import urllib.request
from wolfgym.evaluation.combination_qwen_pilot import QwenTransport,wire_body,decode_reply
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_requests import TransportFailure


def safe_error_chain(exc,secret):
    result=[];seen=set()
    while isinstance(exc,BaseException) and id(exc) not in seen and len(result)<5:
        seen.add(id(exc));message=str(exc)
        if secret:message=message.replace(secret,'[REDACTED]')
        message=re.sub(r'(?i)(Bearer\s+)[^\s\"\']+',r'\1[REDACTED]',message)
        message=re.sub(r'(?i)(https?://)[^/@\s]+:[^/@\s]+@',r'\1[REDACTED]@',message)
        result.append({'type':type(exc).__name__,'errno':getattr(exc,'errno',None),
            'ssl_library':None,'ssl_reason':None,
            'message':message[:500]})
        reason=getattr(exc,'reason',None)
        exc=reason if isinstance(reason,BaseException) else exc.__cause__
    return result


class DiagnosticTransport(QwenTransport):
    def __call__(self,payload):
        encoded=wire_body(payload);identity=digest(payload);dest=self.directory/f'{identity}.json'
        if dest.exists():raise TransportFailure('saved provider response must not be reissued')
        req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=encoded,
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json',
                     'X-Title':'WolfGym bounded train pilot'},method='POST')
        started=time.monotonic()
        try:
            with self.opener.open(req,timeout=120) as response:data=json.load(response)
        except Exception as exc:
            durable_json(self.directory/f'{identity}.failure.json',{'exception_type':type(exc).__name__,
                'http_status':getattr(exc,'code',None),'payload_sha256':identity,
                'elapsed_seconds':time.monotonic()-started,'error_chain':safe_error_chain(exc,self.key),
                'route':'existing system proxy configuration; no fallback or hidden retry'})
            raise TransportFailure('single POST failed; see redacted causal error chain') from None
        message=data.get('choices',[{}])[0].get('message',{})
        safe={'id':data.get('id'),'model':data.get('model'),'provider':data.get('provider'),
            'usage':data.get('usage'),'created':data.get('created'),
            'choices':[{'finish_reason':data.get('choices',[{}])[0].get('finish_reason'),
                        'message':{'content':message.get('content'),'refusal':message.get('refusal')}}]}
        durable_json(dest,safe);return decode_reply(safe)
