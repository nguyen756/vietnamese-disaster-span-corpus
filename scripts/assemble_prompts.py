"""Reconstruct locked P2 locally using exact recovered validation text; no API calls."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def assemble(validation):
    rows=[json.loads(s)for s in Path(validation).read_text(encoding='utf-8-sig').splitlines()if s.strip()]
    by={r['id']:r for r in rows};assert len(by)==len(rows)
    recipe=json.loads((ROOT/'prompts/P2_recipe.json').read_text(encoding='utf-8'));parts=[]
    for piece in recipe['parts']:
        if 'literal'in piece:parts.append(piece['literal']);continue
        row=by[piece['example_id']];text=row['text']
        assert hashlib.sha256(text.encode()).hexdigest()==piece['sentence_sha256']
        annotation=[]
        for start,end,label,occurrence in piece['spans_and_occurrences']:
            substring=text[start:end];positions=[i for i in range(len(text))if text.startswith(substring,i)]
            assert positions[occurrence]==start
            annotation.append([substring,occurrence,label])
        parts.append(json.dumps({'sentence':text,'annotation':annotation},ensure_ascii=False)+piece['newline'])
    prompt=''.join(parts)
    assert hashlib.sha256(prompt.encode()).hexdigest()==recipe['expected_utf8_sha256'],'Prompt byte lock differs'
    return prompt
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    p2=assemble(a.validation);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'P1.txt').write_bytes((ROOT/'prompts/P1.txt').read_bytes())
    (a.output/'P2.txt').write_bytes(p2.encode('utf-8'))
    print('PASS: exact P1/P2 reconstructed; zero API calls. Store generated text outside public repositories.')
