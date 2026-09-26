"""Install the pinned inference adapter into an explicit lmms-eval checkout.

This changes only that checkout's LLaVA adapter, preserving a one-time backup.
It never modifies an installed package discovered indirectly.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lmms-root',type=Path,required=True);a=p.parse_args()
    expected='cb45ac4d4a667ea5ef89c7a148bff69b3489b981'
    actual=subprocess.check_output(['git','-C',str(a.lmms_root),'rev-parse','HEAD'],text=True).strip()
    if actual!=expected: raise RuntimeError('Wrong lmms-eval revision')
    src=Path(__file__).resolve().parent/'compat/lmms_llava.py'
    dst=a.lmms_root/'lmms_eval/models/simple/llava.py';backup=dst.with_suffix('.py.before-synthalign')
    original=subprocess.check_output(['git','-C',str(a.lmms_root),'show','HEAD:lmms_eval/models/simple/llava.py'])
    if dst.read_bytes() not in (original,src.read_bytes()): raise RuntimeError('Refusing to overwrite unrelated local adapter edits')
    if not backup.exists(): backup.write_bytes(original)
    shutil.copyfile(src,dst);print('Installed pinned adapter:',dst)

if __name__=='__main__': main()
