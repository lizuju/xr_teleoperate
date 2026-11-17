#!/usr/bin/env python3
"""Generate phase candidates and a local, self-contained review page; originals stay read-only."""
import argparse
import base64
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop.utils.icecream_task import PHASES, source_digest
from teleop.utils.camera_color import recorded_red_blue_swaps


def propose(rows):
    n = len(rows)
    q = np.asarray([r['states']['body']['qpos'][0] for r in rows])
    grip = np.asarray([r['states']['right_ee']['qpos'][0] for r in rows])
    locks = np.asarray([bool(r['sample'].get('hand_locks', {}).get('right')) for r in rows])
    front = np.median(q[:min(20, n)])
    turned = np.flatnonzero(q < front - np.deg2rad(60))
    lock = np.flatnonzero(locks)
    close = np.flatnonzero(grip < np.max(grip) - .5)
    if not len(turned) or not len(close) or not len(lock):
        return [0, n//5, 2*n//5, 3*n//5, 4*n//5, n], 'No unambiguous turn/grasp/lock; placeholders need review'
    b1, b2, b4 = int(close[0]), int(lock[0]), int(lock[-1] + 1)
    # The end of dispensing is not observable from joints alone. This midpoint is explicitly unreviewed.
    b3 = (b2 + b4) // 2
    points = [0, b1, b2, b3, b4, n]
    if any(a >= b for a, b in zip(points, points[1:])):
        points = [0, n//5, 2*n//5, 3*n//5, 4*n//5, n]
    return points, 'Candidates only: verify cup lift, outlet position, flow stop, hand retraction, and final release in images'


def review(source, output):
    output.mkdir(parents=True, exist_ok=False)
    labels = {'schema': 'r1_icecream_phases_v1', 'phases': PHASES, 'episodes': {}}
    panels = []
    for e in sorted(source.glob('episode_*')):
        manifest = json.loads((e/'episode.json').read_text())
        if manifest.get('status') != 'complete' or manifest.get('outcome') != 'success':
            continue
        swaps = recorded_red_blue_swaps(manifest['info'])
        with (e/'frames.jsonl').open() as f:
            rows = [json.loads(x) for x in f]
        bounds, note = propose(rows)
        entry = {'frames_sha256': source_digest(e/'frames.jsonl'), 'reviewed': False, 'boundaries': bounds, 'note': note}
        labels['episodes'][e.name] = entry
        stamps = [(r['sample']['monotonic_ns']-rows[0]['sample']['monotonic_ns'])/1e9 for r in rows]
        indices = sorted(set(np.linspace(0, len(rows)-1, 36, dtype=int).tolist()+[min(len(rows)-1,b) for b in bounds]))
        pictures=[]
        for index in indices:
            tiles=[]
            for key in ['color_0','color_2','color_3']:
                im=cv2.imread(str(e/rows[index]['colors'][key]))
                if im is None:raise ValueError(f'Missing image in {e.name} frame {index}')
                if key in swaps: im=im[:, :, ::-1]
                im=cv2.resize(im,(240,180))
                tiles.append(im)
            ok,buf=cv2.imencode('.jpg',np.hstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,65])
            pictures.append(f'<figure><img src="data:image/jpeg;base64,{base64.b64encode(buf).decode()}"><figcaption>frame {index} · {stamps[index]:.2f}s</figcaption></figure>')
        fields=' '.join(f'<label>{html.escape(PHASES[i])} start <input type="number" min="0" max="{len(rows)-1}" value="{bounds[i]}" data-bound="{i}"></label>' for i in range(1,5))
        panels.append(f'<section data-episode="{html.escape(e.name)}"><h2>{html.escape(e.name)}</h2><p>{html.escape(note)}</p>{fields}<label><input type="checkbox" data-reviewed> Reviewed all boundaries and task outcome</label><details><summary>Open contact sheet: head / left wrist / right wrist</summary>{"".join(pictures)}</details></section>')
    (output/'candidates.json').write_text(json.dumps(labels,indent=2,ensure_ascii=False)+'\n')
    data=json.dumps(labels).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>R1 phase review</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#f7f8fa;color:#172334}section{padding:20px;background:white;margin:20px 0}label{display:block;margin:10px}input[type=number]{width:90px}figure{display:inline-block;margin:6px}img{max-width:100%}button{padding:14px;position:sticky;top:10px}</style><h1>Review ice-cream phases</h1><p>Original frames are unchanged. Legacy right Dex1 red/blue channels are corrected for this preview. Boundaries are frame indices, including the start and excluding the end. Candidates are not verified labels. Inspect the original recording when these sampled images are insufficient.</p><button id="save">Download reviewed annotations</button>'''+''.join(panels)+'''<script>const data='''+data+''';document.querySelector('#save').onclick=()=>{for(const section of document.querySelectorAll('section')){const e=data.episodes[section.dataset.episode];for(const input of section.querySelectorAll('[data-bound]'))e.boundaries[Number(input.dataset.bound)]=Number(input.value);e.reviewed=section.querySelector('[data-reviewed]').checked;if(e.boundaries.some((v,i,a)=>!Number.isInteger(v)||(i&&v<=a[i-1]))){alert('Boundaries must increase: '+section.dataset.episode);return;}}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='reviewed_phases.json';a.click();URL.revokeObjectURL(a.href);};</script>'''
    (output/'review.html').write_text(page)
    print(output/'review.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();review(a.source,a.output)
