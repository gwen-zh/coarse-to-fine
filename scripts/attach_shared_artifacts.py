"""Link explicitly selected project artifact directories from shared storage.

Creates links only. Never overwrites existing paths or links cache/credentials.
"""
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--names',nargs='+',default=['pdb_training_v2','hierarchical_rigid_pipeline_v1','mixed_hierarchical_generator_v1','mixed_hierarchical_followup_v1','development'])
a=p.parse_args();source=a.source.resolve(strict=True);destination=ROOT/'artifacts';destination.mkdir(exist_ok=True)
for name in a.names:
    if Path(name).name!=name or name.startswith('.') or name in ['cache','hf_cache']:raise ValueError('Select explicit project directories,not caches or hidden files')
    origin=(source/name).resolve(strict=True)
    if not origin.is_dir() or source not in origin.parents:raise ValueError('Expected a directory within the source artifact root')
    target=destination/name
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve()==origin:print('Already linked:',name);continue
        raise FileExistsError('Will not overwrite '+str(target))
    target.symlink_to(origin,target_is_directory=True);print('Linked:',name)
