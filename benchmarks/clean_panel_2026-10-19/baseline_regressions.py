"""Run unchanged acceptance controls against an exported historical candidate.

This is a negative-control diagnostic, not the acceptance suite. Exit zero means
all historical defects were reproduced. Run clean-panel-check for acceptance.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import traceback
import test_repairs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('historical_directory',type=Path)
    args=parser.parse_args()
    def load(name):
        spec=importlib.util.spec_from_file_location('historical_'+name,args.historical_directory/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    test_repairs.panel=load('panel');test_repairs.seal_pilot=load('seal_pilot')
    receipt=test_repairs.ReceiptRepairs()
    controls=[
        ('committed-layout',lambda:test_repairs.SealerLayout('test_committed_directory_with_tiny_fake_environment').debug()),
        ('parse-terminal-receipt',lambda:receipt.exercise(json.JSONDecodeError('malformed','x',0))),
        ('launch-terminal-receipt',lambda:receipt.exercise(FileNotFoundError('launch absent'))),
        ('interruption-terminal-receipt',lambda:receipt.exercise(KeyboardInterrupt('owned interrupt'))),
        ('deleted-artifact-terminal-receipt',lambda:receipt.exercise(FileNotFoundError('deleted artifact'),post=True)),
        ('emitted-workload',lambda:test_repairs.ParserRepairs('test_actual_run_arm_parser_rejects_wrong_native_metadata').debug()),
    ]
    failures=[]
    for name,control in controls:
        try:control()
        except BaseException as error:
            failures.append(name)
            print('EXPECTED HISTORICAL FAILURE',name,type(error).__name__,str(error),flush=True)
            traceback.print_exc()
        else:print('UNEXPECTED HISTORICAL PASS',name,flush=True)
    print(json.dumps(dict(expected_failure_count=len(controls),observed_failures=failures)),flush=True)
    raise SystemExit(0 if len(failures)==len(controls) else 1)


if __name__=='__main__':main()
