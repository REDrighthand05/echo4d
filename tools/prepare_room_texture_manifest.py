"""Prepare original-resolution calibrated training RGB from existing public room archive."""
import argparse,json
from pathlib import Path


def prepare(root, count=24):
    root=Path(root)
    record=json.loads((root/'source/recording.json').read_text())
    association=json.loads((root/'source/association.json').read_text())
    control=json.loads((root/'baseline/report.json').read_text())
    ids=control['training_selection']['expanded_indices']
    if not 2 <= count <= len(ids):
        raise ValueError('Requested RGB count outside retained training set')
    selected=[ids[round(i*(len(ids)-1)/(count-1))] for i in range(count)]
    frames=[]
    for i in selected:
        frame=record['frames'][i].copy()
        frame.pop('depth',None)
        frame.update(color=association['frames'][i]['rgb_path'],rgb_sha256=association['frames'][i]['rgb_sha256'],source_index=i)
        frames.append(frame)
    target=root/'source'/f'texture-{count}.json'
    target.write_text(json.dumps({'intrinsics':record['intrinsics'],'frames':frames},indent=2))
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-root',type=Path,required=True)
    p.add_argument('--count',type=int,default=24)
    args=p.parse_args()
    print(prepare(args.replay_root,args.count))
