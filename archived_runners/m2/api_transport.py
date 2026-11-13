"""List models; optionally send ONE tiny generation. Standard library, no retries."""
import argparse,datetime,getpass,json,os,re,sys,time,uuid,warnings
from decimal import Decimal
from pathlib import Path
from urllib import request,error

ROOT=Path(__file__).resolve().parent
API='https://api.openai.com/v1'
MODEL='gpt-6-luna'
PROMPT='Reply exactly: OK'
OUTPUT_CAP=128
INPUT_PROVISION=16384  # Conservative provision, NOT a measured/API-enforced limit.
MAX_USD=Decimal('0.01')
INPUT_RATE=Decimal('0.10')
OUTPUT_RATE=Decimal('0.50')
RESERVATION_RATE=Decimal('0.25')  # Highest listed 12h cache-write rate, conservative.
EXPIRES=datetime.datetime(2026,11,20,tzinfo=datetime.timezone.utc)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def clean(value,key):
    text=str(value)
    if key:text=text.replace(key,'[REDACTED_KEY]')
    text=re.sub(r'\bsk-[^\s\"\'<>]+','[REDACTED_KEY]',text)
    text=re.sub(r'(?i)bearer\s+[^\s\"\'<>]+','Bearer [REDACTED]',text)
    text=re.sub(r'\b(?:org|proj|user|acct)[-_][A-Za-z0-9_-]+','[REDACTED_ID]',text)
    text=re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}','[REDACTED_EMAIL]',text)
    return text[:3000]
def scrub(value,key):
    if isinstance(value,str):return clean(value,key)
    if isinstance(value,dict):return {clean(k,key):scrub(v,key)for k,v in value.items()}
    if isinstance(value,list):return [scrub(v,key)for v in value]
    return value
class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None
def http(method,path,key,payload=None):
    if path not in ['/models','/responses']:raise ValueError('Unexpected endpoint')
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','User-Agent':'NLP-one-shot-key-test/1'}
    data=None if payload is None else json.dumps(payload).encode('utf-8')
    req=request.Request(API+path,data=data,headers=headers,method=method)
    started=time.perf_counter()
    try:
        # Disable environment proxy routing; only the fixed official HTTPS host.
        with request.build_opener(request.ProxyHandler({}),NoRedirect()).open(req,timeout=90)as response:
            raw=response.read(4*1024*1024);status=response.status;response_headers=response.headers
    except error.HTTPError as e:
        raw=e.read(4*1024*1024);status=e.code;response_headers=e.headers
    except Exception as e:
        return dict(http_status=None,elapsed_seconds=time.perf_counter()-started,data=None,error={'type':type(e).__name__,'code':'network_or_transport_failure','message':'Request failed; no automatic retry. Generation processing/usage may be unknown.'},request_id=None,retry_after=None)
    try:body=json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError,json.JSONDecodeError):body={}
    if not isinstance(body,dict):body={}
    err=body.get('error')or{}
    if not isinstance(err,dict):err={}
    return dict(http_status=status,elapsed_seconds=time.perf_counter()-started,data=scrub(body,key) if 200<=status<300 else None,error=None if 200<=status<300 else {k:clean(err.get(k,'Unavailable; response did not contain a JSON error field'),key)for k in ['type','code','message']},request_id=clean(response_headers.get('x-request-id'),key)if response_headers.get('x-request-id')else None,retry_after=clean(response_headers.get('Retry-After'),key)if response_headers.get('Retry-After')else None)
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def key_from_terminal(use_env):
    if use_env:
        key=os.environ.get('OPENAI_API_KEY','').strip()
    else:
        if not sys.stdin.isatty():raise ValueError('Run in an interactive terminal to enter the key without echo.')
        with warnings.catch_warnings():
            warnings.simplefilter('error',getpass.GetPassWarning)
            key=getpass.getpass('New OpenAI API key (hidden): ').strip()
    if not key:raise ValueError('No key supplied.')
    if any(c.isspace()for c in key):raise ValueError('Key contains whitespace; no request sent.')
    return key
def preflight():
    reserve=(Decimal(INPUT_PROVISION)*RESERVATION_RATE+Decimal(OUTPUT_CAP)*OUTPUT_RATE)/Decimal(1000000)
    if datetime.datetime.now(datetime.timezone.utc)>=EXPIRES:raise ValueError('Saved price evidence expired; recheck official pricing before generation.')
    if reserve>MAX_USD:raise ValueError('Conservative request provision exceeds the configured ceiling.')
    return dict(model=MODEL,prompt=PROMPT,reasoning_effort='none',max_output_tokens=OUTPUT_CAP,service_tier='default',input_token_provision=INPUT_PROVISION,input_provision_is_measured=False,input_provision_is_API_enforced=False,conservative_request_provision_usd=str(reserve),allowed_provision_usd=str(MAX_USD),input_usd_per_million=str(INPUT_RATE),output_usd_per_million=str(OUTPUT_RATE),reservation_input_usd_per_million=str(RESERVATION_RATE),pricing_checked_date='2026-11-12',pricing_sources=['https://developers.openai.com/api/docs/models/gpt-6-luna','https://developers.openai.com/api/docs/pricing'],generation_attempts_allowed=1,retries=0,fallbacks=0,tools=False,billing_or_account_limit_changed=False)
def claim(path,provision):
    try:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:raise ValueError('Generation was already claimed in this folder. No second attempt, retry or ledger reset is performed.')
    with os.fdopen(fd,'w',encoding='utf-8')as f:
        json.dump(dict(claimed_at_utc=utc(),state='claimed_before_POST; preserve if failure or interruption',reserved_usd=provision['conservative_request_provision_usd'],provision=provision),f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def generation(key,report_dir):
    provision=preflight()
    # Permanent claim written and flushed BEFORE a possible paid request.
    claim(ROOT/'GENERATION_ATTEMPT_CLAIM.json',provision)
    save(report_dir/'generation_preflight.json',provision)
    payload=dict(model=MODEL,input=PROMPT,max_output_tokens=OUTPUT_CAP,reasoning={'effort':'none'},service_tier='default',tools=[],store=False)
    result=http('POST','/responses',key,payload)
    data=result.pop('data');result.update(requested_model=MODEL,attempts=1,actual_billing_receipt=None)
    if data is None:
        result.update(generation_succeeded=False,usage=None,calculated_cost_usd=None,cost_status='unresolved; do not assume zero charge')
    else:
        usage=data.get('usage');text=''.join(c.get('text','')for out in data.get('output',[])if out.get('type')=='message'for c in out.get('content',[])if c.get('type')=='output_text')
        cost=None
        if isinstance(usage,dict)and type(usage.get('input_tokens'))is int and type(usage.get('output_tokens'))is int:
            cost=(Decimal(usage['input_tokens'])*INPUT_RATE+Decimal(usage['output_tokens'])*OUTPUT_RATE)/Decimal(1000000)
        # Only synthetic test text is sent/returned. Still redact defensively.
        result.update(generation_succeeded=data.get('status')=='completed',response_status=data.get('status'),returned_model=clean(data.get('model'),key),returned_text=clean(text,key),usage=scrub(usage,key),calculated_cost_usd=str(cost)if cost is not None else None,cost_status='standard-rate token calculation, not a billing receipt; assumes uncached text input/output rates and no cache-write charges'if cost is not None else 'unresolved; usage unavailable, do not assume zero charge',incomplete_details=scrub(data.get('incomplete_details'),key))
    save(report_dir/'generation_result.json',result)
    return result
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true',help='After listing models, attempt one fixed tiny generation using the permanent claim.')
    parser.add_argument('--env-key',action='store_true',help='Explicitly use OPENAI_API_KEY; default prompts for the NEW key with hidden input.')
    args=parser.parse_args()
    try:key=key_from_terminal(args.env_key)
    except (ValueError,getpass.GetPassWarning,EOFError,KeyboardInterrupt)as e:
        print('STOP: '+str(e));return 2
    report_dir=ROOT/'reports'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);report_dir.mkdir(parents=True,exist_ok=False)
    result=http('GET','/models',key);data=result.pop('data')
    ids=sorted({r['id']for r in data.get('data',[])if isinstance(r.get('id'),str)})if data else []
    result.update(authentication_passed=result['http_status']==200,model_count=len(ids),generation_requested=args.generate,generation_access_proven=False,billing_or_credits_proven=False)
    save(report_dir/'authentication_result.json',result);save(report_dir/'models.json',dict(model_ids=ids,listed_at_utc=utc(),generation_access_not_proven_by_listing=True))
    print(json.dumps(result,indent=2))
    if result['http_status']!=200:
        print('STOP. Model listing failed. No generation attempted. Reports: '+str(report_dir));return 1
    print('Listed GPT/chat models:')
    for name in ids:
        if name.lower().startswith(('gpt','chatgpt','o1','o3','o4')):print('  '+clean(name,key))
    if args.generate:
        if MODEL not in ids:
            print('STOP: '+MODEL+' is not listed; no generation or alternate-model fallback attempted.')
        else:
            try:result=generation(key,report_dir);print(json.dumps(result,ensure_ascii=False,indent=2))
            except ValueError as e:print('STOP: '+clean(str(e),key))
    print('Reports: '+str(report_dir))
    print('No annotation pilot/full study was launched. Do not share your key; share the report files.')
    return 0
if __name__=='__main__':raise SystemExit(main())
