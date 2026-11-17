#!/usr/bin/env python3
"""Record the observed outcome and link human corrections; never relabel policy commands as demonstrations."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def record_result(run, outcome, failure_phase, reason, corrections):
    summary = run/'summary.json'
    if not summary.is_file():
        raise ValueError('Policy run must be stopped and have summary.json')
    links=[]
    for path in corrections:
        m=json.loads((path/'episode.json').read_text())
        if m.get('status')!='complete' or m.get('outcome')!='success':
            raise ValueError('Corrections must be completed human demonstrations marked success')
        links.append({'episode':str(path.resolve()),'manifest_sha256':hashlib.sha256((path/'episode.json').read_bytes()).hexdigest()})
    document={'schema':'r1_policy_result_v1','time':time.time(),'outcome':outcome,'failure_phase':failure_phase,
              'reason':reason,'corrections':links,'run_summary_sha256':hashlib.sha256(summary.read_bytes()).hexdigest()}
    with (run/'results.jsonl').open('a') as f:f.write(json.dumps(document,ensure_ascii=False)+'\n')
    return document


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--outcome',choices=['success','failure','intervention'],required=True)
    p.add_argument('--failure-phase',choices=['grasp_cup','turn_deliver','dispense','close_retract','turn_place'])
    p.add_argument('--reason',required=True);p.add_argument('--correction',type=Path,action='append',default=[])
    a=p.parse_args();print(json.dumps(record_result(a.run,a.outcome,a.failure_phase,a.reason,a.correction),ensure_ascii=False))
