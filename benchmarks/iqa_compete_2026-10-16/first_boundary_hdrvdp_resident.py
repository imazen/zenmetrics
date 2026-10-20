#!/usr/bin/env python3
"""Prepare separate official HDR resident-parameter fixtures; never execute them.

Metric trees and evaluator bodies are byte-preserved. A new path shim serves
the original spectral loader's preloaded outputs by exact request key only.
Run the separately admitted preflight and original/resident controls before
using any prepared warm or first evaluator timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


LOADER_SHA = 'f51b19ae6a874bf5474a0bee367d9564fac84b091b9d343bc7c7a6e72e6ca560'
TABLES = {
    'hdrvdp': ('emission_spectra_ccfl-lcd.csv',
               'log_cone_smith_pokorny_1975.csv', 'cie_scotopic_lum.txt'),
    'hdrvdp3': ('emission_spectra_led-lcd-srgb.csv',
                'log_cone_smith_pokorny_1975.csv', 'cie_scotopic_lum.txt', 'ciexyz31.csv'),
}

# New fixture code only. No original M-file body is copied into this owner.
CACHE = r'''function [lambda, R] = first_boundary_spectral_cache(action, varargin)
  persistent sealed = false;
  persistent keys wavelengths responses;
  lambda = []; R = [];
  switch action
    case 'install'
      if sealed
        error('first_boundary:cache_sealed', 'Spectral cache is already sealed');
      end
      assert(numel(varargin)==3);
      incoming = varargin{1}; wl = varargin{2}; rr = varargin{3};
      assert(iscell(incoming) && iscell(wl) && iscell(rr));
      assert(!isempty(incoming) && numel(incoming)==numel(wl) && numel(wl)==numel(rr));
      assert(numel(unique(incoming))==numel(incoming));
      for k=1:numel(incoming)
        assert(ischar(incoming{k}) && rows(incoming{k})==1 && !isempty(incoming{k}));
        assert(isa(wl{k},'double') && isa(rr{k},'double'));
        assert(isreal(wl{k}) && isreal(rr{k}) && all(isfinite(wl{k}(:))) && all(isfinite(rr{k}(:))));
        assert(rows(wl{k})==1 && rows(rr{k})==numel(wl{k}));
      end
      keys = incoming; wavelengths = wl; responses = rr; sealed = true;
    case 'get'
      assert(sealed && numel(varargin)==1 && ischar(varargin{1}) && rows(varargin{1})==1);
      index = find(strcmp(keys,varargin{1}));
      if numel(index)!=1
        error('first_boundary:spectral_cache_miss', 'Unbound spectral request: %s', varargin{1});
      end
      lambda = wavelengths{index}; R = responses{index};
    otherwise
      error('first_boundary:cache_action', 'Unsupported spectral cache action');
  end
end
'''
SHIM = '''function [lambda, R] = load_spectral_resp(file_name)
  [lambda, R] = first_boundary_spectral_cache('get',file_name);
end
'''


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('source/input binding changed: ' + path)


def exact_arrays(left, right):
    return (f'assert(strcmp(class({left}),class({right})) && '
            f'isequal(size({left}),size({right})) && '
            f'isequal(num2hex({left}(:)),num2hex({right}(:))));')


def topology(metric, tree):
    v3 = metric == 'hdrvdp3'
    if metric not in TABLES:
        raise ValueError('official base HDR routes only')
    data = tree/'data' if v3 else tree
    physical = [data/name for name in TABLES[metric]]
    keys = [str(p) if v3 and p.name != 'ciexyz31.csv' else p.name for p in physical]
    loader = tree/'utils/load_spectral_resp.m' if v3 else tree/'load_spectral_resp.m'
    return loader, physical, keys


def bootstrap(metric, tree, fixture, missing=None):
    loader, physical, keys = topology(metric, tree)
    lines = [f'assert(strcmp(which("load_spectral_resp"),{quote(loader)}));',
             'cache_keys={' + ','.join(quote(key) for key in keys) + '};',
             'cache_files={' + ','.join(quote(p) for p in physical) + '};',
             'cache_wl=cell(size(cache_keys)); cache_rr=cell(size(cache_keys));',
             'for k=1:numel(cache_keys)',
             '  [cache_wl{k},cache_rr{k}]=load_spectral_resp(cache_files{k});',
             'end',
             f'addpath({quote(fixture)},"-begin");']
    if missing is None:
        lines += ["first_boundary_spectral_cache('install',cache_keys,cache_wl,cache_rr);"]
    else:
        if missing not in range(len(keys)):
            raise ValueError('invalid omitted cache key')
        lines += [f'keep=true(size(cache_keys));keep({missing+1})=false;',
                  "first_boundary_spectral_cache('install',cache_keys(keep),cache_wl(keep),cache_rr(keep));"]
    lines += ['clear load_spectral_resp; rehash;',
              f'assert(strcmp(which("load_spectral_resp"),{quote(fixture/"load_spectral_resp.m")}));',
              f'assert(strcmp(which("first_boundary_spectral_cache"),{quote(fixture/"first_boundary_spectral_cache.m")}));',
              "fprintf('PARAMETER_CACHE_SEALED keys=%d original_helper=%s\\n'," + str(len(keys)-(missing is not None)) + ',' + quote(loader) + ');',
              'fflush(stdout);']
    return '\n'.join(lines)+'\n'


def array_control(metric, tree, fixture):
    # Fresh original-helper reads occur before installing the shim. Afterwards
    # every lookup must return identical class, shape and IEEE-754 bits.
    text = bootstrap(metric, tree, fixture)
    point = f'addpath({quote(fixture)},"-begin");\n'
    fresh = ('fresh_wl=cell(size(cache_keys));fresh_rr=cell(size(cache_keys));\n'
             'for k=1:numel(cache_keys)\n'
             '  [fresh_wl{k},fresh_rr{k}]=load_spectral_resp(cache_files{k});\n'
             'end\n')
    if text.count(point) != 1:
        raise ValueError('array bootstrap insertion is ambiguous')
    text = text.replace(point, fresh+point)
    text += 'for k=1:numel(cache_keys)\n  [wl,rr]=load_spectral_resp(cache_keys{k});\n'
    text += exact_arrays('wl','fresh_wl{k}')+'\n'+exact_arrays('rr','fresh_rr{k}')+'\n'
    text += ('  wl(1)=wl(1)+1;rr(1)=rr(1)+1;\n'
             '  [wl,rr]=load_spectral_resp(cache_keys{k});\n')
    text += exact_arrays('wl','fresh_wl{k}')+'\n'+exact_arrays('rr','fresh_rr{k}')+'\n'
    text += ("  fprintf('PARAMETER_ARRAY_EXACT key=%s rows=%d cols=%d\\n',cache_keys{k},rows(rr),columns(rr));\n"
             'end\n'
             "try; first_boundary_spectral_cache('install',cache_keys,cache_wl,cache_rr); error('first_boundary:expected_seal_refusal','Replacement accepted');\n"
             "catch err; assert(strcmp(err.identifier,'first_boundary:cache_sealed')); end\n"
             "try; load_spectral_resp('first-boundary-unbound-table'); error('first_boundary:expected_key_refusal','Unknown key accepted');\n"
             "catch err; assert(strcmp(err.identifier,'first_boundary:spectral_cache_miss')); end\n"
             "try; first_boundary_spectral_cache('reset'); error('first_boundary:expected_action_refusal','Reset accepted');\n"
             "catch err; assert(strcmp(err.identifier,'first_boundary:cache_action')); end\n"
             "fprintf('PARAMETER_ARRAY_COMPLETE keys=%d passed=1\\n',numel(cache_keys));fflush(stdout);\n")
    return text


def rewrite(source, replacements):
    # Longest first: data/utils paths contain their official-tree prefix.
    for old, new in sorted(replacements.items(), key=lambda row: -len(row[0])):
        source = source.replace(old, new)
    return source


def split_adapter(text):
    point = text.find('f=fopen(')
    if point < 0 or text.count('clock=tic();') != 1:
        raise ValueError('unexpected original evaluator wrapper layout')
    setup, evaluation = text[:point], text[point:]
    start = evaluation.index('clock=tic();\n')+len('clock=tic();\n')
    end = evaluation.index('elapsed=toc(clock)', start)
    body = evaluation[start:end]
    if 'load_spectral_resp' in body or body.count('result=hdrvdp') != 1:
        raise ValueError('unexpected original evaluator body')
    return setup, evaluation, body


def prepare(parent, resize, output):
    parent, resize = parent.resolve(strict=True), resize.resolve(strict=True)
    index_path = parent/'index.json'
    index = json.loads(index_path.read_text())
    verify(index['bindings'])
    verify(index['plans'])
    plans = [json.loads(Path(path).read_text()) for path in sorted(index['plans'])]
    cases = [case for plan in plans for case in plan['cases']]
    if len(cases)!=104 or len({c['id'] for c in cases})!=104:
        raise ValueError('complete unchanged current40 104-case inventory required')
    if {c['threads'] for c in cases}!={1,28}:
        raise ValueError('one/twenty-eight budgets required')
    output.mkdir()
    root = output.resolve()
    fixture = root/'fixture'; fixture.mkdir()
    (fixture/'first_boundary_spectral_cache.m').write_text(CACHE)
    (fixture/'load_spectral_resp.m').write_text(SHIM)
    (root/'execution-cwd').mkdir()
    bindings = {str(index_path):sha(index_path), **index['plans']}
    helpers = root/'helpers';helpers.mkdir()
    for name in ('first_boundary_hdrvdp_resident.py', 'first_boundary_hdrvdp_resident_run.py',
                 'first_boundary_hdrvdp_admitted.py'):
        shutil.copy2(Path(__file__).with_name(name),helpers/name)
    replacements = {}
    for metric, tag in (('hdrvdp','official-v2'),('hdrvdp3','official-v3')):
        original = next(Path(json.loads(Path(c['reference_config']).read_text())['tree'])
                        for c in cases if c['metric']==metric)
        dest = root/tag
        # Only source-bound files from the original tree are copied, preserving
        # every byte. Assets stay private; no original body enters this script.
        for path, expected in index['bindings'].items():
            if Path(path).is_relative_to(original):
                new = dest/Path(path).relative_to(original)
                new.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,new)
                if sha(new)!=expected:
                    raise ValueError('original source copy changed')
        replacements[str(original)]=str(dest)
        if sha(topology(metric,dest)[0])!=LOADER_SHA:
            raise ValueError('original spectral helper differs from reviewed source')
    config = json.loads(Path(cases[0]['reference_config']).read_text())
    geomean = Path(config['geomean'])
    statistics = root/'statistics'; statistics.mkdir()
    shutil.copy2(geomean,statistics/geomean.name)
    replacements[str(geomean.parent)]=str(statistics)
    overlay = root/'resize2201';(overlay/'private').mkdir(parents=True)
    for name, expected in (
        ('imresize.m','c358d5381f325c5ab58c7eb0c3ccc8acc2141bc70634c66cc516541b823a15c8'),
        ('private/interp_method.m','bcd3156239d6d4dccb626108a642afead1b4c60b400fed8b11d73ef5edcc6beb')):
        if sha(resize/name)!=expected:
            raise ValueError('unchanged resize source required')
        shutil.copy2(resize/name,overlay/name)
    manifest_cases = {}
    (root/'variants').mkdir()
    for variant in ('system212','resize2201'):
        variant_root=root/'variants'/variant;variant_root.mkdir()
        scores=[];preflight=[]
        for old in cases:
            metric=old['metric'];tree=root/('official-v3' if metric=='hdrvdp3' else 'official-v2')
            text=Path(old['canonical']['command'][-1]).read_text()
            text=rewrite(text,replacements)
            setup,evaluation,body=split_adapter(text)
            helper=topology(metric,tree)[0]
            setup+=f'cd({quote(root/"execution-cwd")});\n'
            if variant=='resize2201':
                setup+=f'addpath({quote(overlay)},"-begin");\n'
            resize_file=overlay/'imresize.m' if variant=='resize2201' else Path('/usr/share/octave/packages/image-2.12.0/imresize.m')
            setup+=f'assert(strcmp(which("imresize"),{quote(resize_file)}));\n'
            setup+=f'assert(strcmp(which("load_spectral_resp"),{quote(helper)}));\n'
            setup+='fprintf("RESIDENT_RUNTIME_BIND imresize=%s loader=%s\\n",which("imresize"),which("load_spectral_resp"));fflush(stdout);\n'
            # The original body between tic and toc is unchanged. New scalar
            # hexadecimal observations and cache checks occur after toc.
            timed=evaluation.replace('fflush(stdout);',"fflush(stdout);fprintf('SCORE_HEX %s\\n',num2hex(score));fflush(stdout);")
            entry={k:v for k,v in old.items() if k not in ('canonical','direct-control','reference_config','scope')}
            entry.update(variant=variant,qualification='unmatched_scope',matched_scope=False,
                evaluator_body_sha256=hashlib.sha256(body.encode()).hexdigest(),arms={})
            for role in ('default','resident'):
                path=variant_root/(old['id']+'-'+role+'.m')
                boot=bootstrap(metric,tree,fixture) if role=='resident' else ''
                suffix=''
                if role=='resident':
                    suffix='for k=1:numel(cache_keys)\n[wl,rr]=load_spectral_resp(cache_keys{k});\n'
                    suffix+=exact_arrays('wl','cache_wl{k}')+'\n'+exact_arrays('rr','cache_rr{k}')+'\nend\n'
                    suffix+="fprintf('PARAMETER_CACHE_UNCHANGED passed=1\\n');fflush(stdout);\n"
                complete=setup+boot+timed+suffix
                if complete.count(body)!=evaluation.count(body):
                    raise ValueError('evaluator arithmetic/body changed')
                path.write_text(complete)
                entry['arms'][role] = dict(command=[old['canonical']['command'][0],
                    '--no-gui','--no-init-file','--no-history','--quiet',str(path)],
                    parameter_io='outside clock, original-helper outputs' if role=='resident' else 'inside clock, unchanged original loader')
            scores.append(entry)
            # Each version needs independent preflight on only the photo/one
            # thread/first identity; full panel coverage belongs to score gates.
            if old['panel']=='photo512' and old['phase']=='first' and old['threads']==1:
                path=variant_root/(metric+'-array-control.m')
                path.write_text(setup+array_control(metric,tree,fixture))
                preflight.append(dict(id=metric+'-arrays',metric=metric,kind='arrays',keys=len(TABLES[metric]),
                    command=[old['canonical']['command'][0],'--no-gui','--no-init-file','--no-history','--quiet',str(path)]))
                for missing,key in enumerate(topology(metric,tree)[2]):
                    path=variant_root/(metric+f'-missing-{missing}.m')
                    negative=setup+bootstrap(metric,tree,fixture,missing)
                    negative+='cache_miss_seen=false;\ntry\n'+evaluation
                    negative+=("catch err\nassert(strcmp(err.identifier,'first_boundary:spectral_cache_miss'));\n"
                               f'assert(strcmp(err.message,["Unbound spectral request: ",{quote(key)}]));\n'
                               'cache_miss_seen=true;\nend\nassert(cache_miss_seen);\n'
                               f"fprintf('PARAMETER_MISSING_EXACT index={missing} passed=1\\n');fflush(stdout);\n")
                    path.write_text(negative)
                    preflight.append(dict(id=metric+f'-missing-{missing}',metric=metric,kind='missing',
                        key=key,index=missing,command=[old['canonical']['command'][0],'--no-gui','--no-init-file','--no-history','--quiet',str(path)]))
        manifest_cases[variant]=dict(scores=scores,preflight=preflight)
    for path in sorted(root.rglob('*')):
        if path.is_file():bindings[str(path)]=sha(path)
    for old in cases:
        for role in ('reference','distorted'):
            bindings[old[role]]=index['bindings'][old[role]]
        bindings[old['native_config']]=index['bindings'][old['native_config']]
    binary=Path(cases[0]['binary'])
    bindings[str(binary)]=index['bindings'][str(binary)]
    runtime=Path(cases[0]['canonical']['command'][0]);bindings[str(runtime)]=index['bindings'][str(runtime)]
    common=dict(schema=1,bindings=bindings,parent_index_sha256=sha(index_path),
        rust_source=index['source'],binary_sha256=index['binary_sha256'],execution_ready=False,
        matched_scope=False,qualification='unmatched_scope',qualified_timing_cells=0,
        expected_default_resident_exact=True,expected_control_order=['default','resident','resident','default'],
        runtime_roots=['/usr/share/octave/6.4.0','/usr/share/octave/packages/image-2.12.0',
            '/usr/lib/x86_64-linux-gnu/octave/6.4.0','/usr/lib/x86_64-linux-gnu/octave/packages/image-2.12.0',
            str(root/'official-v3'),str(statistics),str(fixture)],
        octave=str(runtime),seconds_per_stage=1100,required_before_execution=[
            'Root code review and shared heavy admission',
            'Root validation of retained golden/runtime evidence',
            'Root exhaustive source-verification receipt before native inference'],
        note='Controls only; no native scorer is invoked by the resident control runner. No absent v3 bound is supplied.')
    for variant,rows in manifest_cases.items():
        path=root/(variant+'.plan.json')
        write(path,dict(common,variant=variant,cases=rows['scores'],preflight=rows['preflight']))
    result=dict(schema=1,bindings=bindings,plans={str(p):sha(p) for p in root.glob('*.plan.json')},
        cases_per_variant=104,preflight_per_variant=9,variants=list(manifest_cases),
        execution_ready=False,matched_scope=False,qualified_timing_cells=0)
    write(root/'index.json',result)
    for path in root.rglob('*'):
        if path.is_file():path.chmod(0o444)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','resize','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args()
    result=prepare(args.parent,args.resize,args.output)
    print(json.dumps({k:result[k] for k in ('cases_per_variant','preflight_per_variant','variants','execution_ready')}))


if __name__=='__main__':main()
