"""Receive the inspected C increment without overwriting historical D evidence."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import hashlib
import json
import shutil
import argparse

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT/'outputs/D/history/reference_20261015'
RECEIVED = ROOT/'outputs/C/received'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    source=parser.parse_args().archive
    if (ROOT/'outputs/C/local_integration.json').exists():
        raise SystemExit('C already received; preserve intake record rather than repeat extraction')
    if any((ROOT/f'outputs/D/{name}').exists() for name in ['freeze.json','test_started.json']):
        raise SystemExit('Existing final-stage record: retain and review before changes')
    HISTORY.mkdir(parents=True, exist_ok=False)
    for rel in ['outputs/D/validation','outputs/D/figures']:
        shutil.copytree(ROOT/rel,HISTORY/Path(rel).name)
    for rel in ['config/d_protocol.json','notebooks/D_evaluation.ipynb',
                'output/pdf/LIANG_SHUANG_D_Evaluation_Report.pdf','notes/D_HANDOFF.md',
                'notes/D_VIDEO_SCRIPT.md','START_D.md','scripts/build_d_artifacts.py',
                'scripts/validate_d.py','outputs/D/qa.json','outputs/D/notebook_validation.json']:
        destination=HISTORY/'original_files'/rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,destination)
    RECEIVED.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,RECEIVED/'C_Pipeline_original.zip')
    files=[]
    with ZipFile(source) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name=PurePosixPath(entry.filename)
            if name.is_absolute() or '..' in name.parts or name.parts[0]!='C_Pipeline' or entry.file_size>1000000:
                raise ValueError('Unexpected archive member')
            rel=PurePosixPath(*name.parts[1:]).as_posix()
            raw=archive.read(entry)
            # Supplied evidence is preserved separately from this machine's reruns.
            target=RECEIVED/rel if rel.startswith('outputs/C/') else ROOT/rel
            if rel=='Integration_Guide.txt':
                target=ROOT/'notes/C_Integration_Guide_received.txt'
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
            files.append({'archive_path':rel,'local_path':target.relative_to(ROOT).as_posix(),
                          'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    config=ROOT/'config/d_protocol.json'
    protocol=json.loads(config.read_text(encoding='utf-8'))
    protocol.update(status='C_integrated_pending_B_and_human_review',
                    pipeline_module='src.preprocessing',
                    pipeline_provenance='C_Pipeline(1).zip supplied by user; source copied unchanged and locally verified; human review pending')
    config.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    record={'received_date_local':'2026-10-16','archive_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'pipeline':'src.preprocessing','source_modified':False,'files':files,
            'historical_D_results':str(HISTORY.relative_to(ROOT)).replace('\\','/'),
            'environment_policy':'Rerun all four configurations in the existing D environment, not mixing supplied C results',
            'missing_from_increment':['analysis.ipynb','scripts/build_c_delivery.py','scripts/execute_c_notebooks.py'],
            'real_test_evaluated':False,'human_review':'pending'}
    (ROOT/'outputs/C/local_integration.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Preserved old D artifacts; received C source unchanged; configured src.preprocessing.')


if __name__=='__main__':
    main()
