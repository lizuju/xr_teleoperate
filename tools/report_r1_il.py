#!/usr/bin/env python3
"""Audit temporal coverage, episode leakage and human-scored policy outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from teleop.utils.act_contract import validate_contract
from teleop.utils.act_dataset import action_chunk_indices


def dataset_report(root,chunk_size=40):
    d=json.loads((root/'dataset.json').read_text());validate_contract(d)
    sources={};splits={};hist=Counter()
    source_hashes={s["source_id"]:s["frames_sha256"] for s in d["sources"]}
    seen_hashes={}
    for split,ids in d['splits'].items():
        episodes=[e for e in d['episodes'] if e['episode_id'] in ids]
        source_ids={e['source_id'] for e in episodes}
        for source in source_ids:
            if source in sources:raise ValueError('Source episode leaks between splits')
            sources[source]=split
            digest=source_hashes[source]
            if digest in seen_hashes and seen_hashes[digest]!=split:
                raise ValueError('Identical source contents leak between splits')
            seen_hashes[digest]=split
        anchors=0
        for e in episodes:
            with h5py.File(root/e['file'],'r') as h:
                times=h['source/action_anchor_ns'][:];phase=h['source/action_phase'][:]
                for row in h['source/anchor_index'][:]:
                    index,pad=action_chunk_indices(times,int(row),chunk_size,d['fps'],int(h["source/monotonic_ns"][-1]))
                    pad |= phase[index]!=phase[row]
                    hist[int((~pad).sum())]+=1
                anchors+=len(h['observations/qpos'])
        splits[split]={'original_episodes':len(source_ids),'segments':len(episodes),'anchors':anchors}
    return {'dataset':str(root),'splits':splits,'chunk_size':chunk_size,
            'usable_action_steps_histogram':dict(sorted(hist.items())),
            'full_horizon_anchors':hist[chunk_size],'anchor_count':sum(hist.values()),
            'imu_used':d['imu_used'],'independent_unit':'original episode, same recording day; not cross-scene proof'}


def policy_results(root):
    groups={}
    for result in sorted(root.glob('*/results.jsonl')):
        run=result.parent
        manifest=json.loads((run/'run.json').read_text())
        if manifest.get('mode')!='execute':continue
        records=[json.loads(line) for line in result.read_text().splitlines() if line.strip()]
        if not records:continue
        final=records[-1]
        group=groups.setdefault(manifest['checkpoint_sha256'],{'trials':0,'success':0,'failure':0,'intervention':0,'failure_phases':Counter()})
        group['trials']+=1;group[final['outcome']]+=1
        if final.get('failure_phase'):group['failure_phases'][final['failure_phase']]+=1
    for g in groups.values():
        g['success_rate']=g['success']/g['trials']
        g['failure_phases']=dict(g['failure_phases'])
    return groups


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path);p.add_argument('--policy-runs',type=Path)
    p.add_argument('--chunk-size',type=int,default=40);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report={}
    if a.dataset:report['dataset']=dataset_report(a.dataset,a.chunk_size)
    if a.policy_runs:report['policy_results']=policy_results(a.policy_runs)
    if not report:p.error('Pass --dataset and/or --policy-runs')
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
