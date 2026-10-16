"""Create a portable review bundle; exclude virtualenv, Git metadata and course PDF."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'outputs/D/delivery_manifest.json'
paths=[]
for folder in ['config','data','notebooks','notes','outputs','scripts','src','output/pdf']:
    paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo'])
for name in ['README.md','START_D.md','analysis.ipynb','C_preprocessing_tests.py','THIRD_PARTY_NOTICES.md','requirements.in','requirements.lock.txt','requirements.D.lock.txt','requirements.C.lock.txt','.gitignore','.gitattributes']:
    paths.append(ROOT/name)
paths=sorted(set(p for p in paths if p != manifest))
record={'time_utc':datetime.now(timezone.utc).isoformat(),
        'owner':'LIANG SHUANG / D','A_base_commit':'b4c4ff1be75a6e48baacd920f405982b20399352',
        'stage':'C formally integrated and locally verified; validation complete; B evidence, human review and final test pending',
        'excluded':['.venv','.git','.local','tmp','original course manual'],
        'files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}}
manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
archive=ROOT.parent/'LIANG_SHUANG_D_C_Integrated_20261016.zip'
with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=7) as z:
    for path in paths+[manifest]:
        z.write(path,arcname='ca6003-bank-marketing/'+path.relative_to(ROOT).as_posix())
with ZipFile(archive) as z:
    assert z.testzip() is None
    for rel,info in record['files'].items():
        assert hashlib.sha256(z.read('ca6003-bank-marketing/'+rel)).hexdigest()==info['sha256']
print(json.dumps({'zip':str(archive),'files':len(paths)+1,'bytes':archive.stat().st_size,'verified':True},ensure_ascii=False))
