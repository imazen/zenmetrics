"""Invented checkpoint projections only; no training or real target reads."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import e36_fit_contract as e
import qualified_fit_contract as q
import test_strict_fit_harvest
import fit_cell_exec
import pack_fit_program


class E36Contract(unittest.TestCase):
    def test_diagnostic_smoke_requires_distinct_frozen_variant(self):
        base=dict(epochs=120,pairs_per_epoch=50000,selected_epoch=119,
            e36=dict(arm='tv10-c',contract_sha256='b'*64,base_filter_sha256='c'*64,trainer_sha256='a'*64))
        argv=['v2_confirm_fit.py','--e36-arm','tv10-c','--e36-tv-contract-sha256','b'*64,'--e36-base-filter-sha256','c'*64]
        files={'bin/zensim_mlp_train':'a'*64}
        smoke=argv+['--local-smoke-budget','2:128']
        with self.assertRaises(ValueError):e.bind(smoke,base,files)
        diagnostic=base|dict(e36_memory_smoke=copy.deepcopy(e.SMOKE))
        e.bind(smoke,diagnostic,files)
        with self.assertRaises(ValueError):e.bind(argv,diagnostic,files)
        for field,bad in [('epochs',3),('pairs_per_epoch',129),('wall_cap_sec',901),('installable',True)]:
            changed=copy.deepcopy(diagnostic);changed['e36_memory_smoke'][field]=bad
            with self.assertRaises(ValueError):e.bind(smoke,changed,files)
        with self.assertRaises(ValueError):e.bind(smoke[:-1]+['1:128'],diagnostic,files)
        # A diagnostic policy does not excuse a different final scientific recipe.
        with self.assertRaises(ValueError):e.bind(smoke,diagnostic|dict(selected_epoch=118),files)

    def test_mask_derivation_preserves_existing_checkpoint_checks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture = test_strict_fit_harvest.StrictHarvest(); cell = fixture.cell(root)
            result = json.loads(cell.files['result.json'])
            expected = fixture.contract | dict(schema='e36-research-fit-contract-v1', route='kadid',
                seed_index=0, init_seed=1101, sample_seed=101, execution_contract='registered-fit',
                table_admission=fixture.contract['routes']['kadid'], train_weights={'kadid': {'human': 32.}})
            roles = dict(loss_mode='Rank', n_features=1825, rows=2, train_w=32., val_w=0., within_ref=True)
            expected['input_contracts'] = {'kadid': {'human': roles}}
            result['train_weights'] = {'human': 32.}
            projection = dict(source_table_sha256='d'*64, source_manifest_sha256='e'*64,
                source_row_keys_sha256='a'*64, source_row_selection_sha256='b'*64,
                keys_sha256='4'*64, row_keys_sha256='5'*64, row_selection_sha256='6'*64,
                rows_in=2, rows_kept=1, targets_changed=0, e36_base_filter_sha256='7'*64)
            wanted = dict(removed=1, manifest=projection)
            expected['e36'] = dict(arm='tv1-c', manifest_sha256='8'*64, contract_sha256='9'*64,
                trainer_sha256='a'*64, mean_pool_weight=1., protocol_source_sha256='b'*64, base_filter_sha256='7'*64,
                curated_tables={'d'*64: wanted})
            child = projection | dict(table_sha256='c'*64)
            child_sha = hashlib.sha256((json.dumps(child, indent=2)+'\n').encode()).hexdigest()
            record = dict(name='human', source_table_sha256='d'*64, removed=1, targets_changed=0,
                table_sha256='c'*64, manifest_sha256=child_sha, stored_manifest=child)
            result['selection']['e36_base_filter'] = dict(schema='e36-applied-base-filter-v1',
                plan_sha256='7'*64, targets_changed=0, exact_group_weights_preserved=True, groups=[record])
            admitted = copy.deepcopy(expected['table_admission'])
            admitted[0].update({k: child[k] for k in ('table_sha256','keys_sha256','row_keys_sha256','row_selection_sha256')})
            admitted[0]['manifest_sha256'] = child_sha
            result['selection']['strict_table_admission'] = admitted
            decoded = fixture.decoded
            decoded['repro']['inputs'][0].update(roles | dict(rows=1, sha256='c'*64))
            decoded['repro'].update(effective_minibatch=1, target_column='human_score', target_scale=1.)
            decoded['repro']['e36_tv'] = dict(schema='e36-effective-tv-v1', protocol_build='e36_tv_v1',
                cfg_e36_tv=True, manifest_sha256='8'*64, contract_sha256='9'*64, binary_sha256='a'*64,
                embedded_protocol_source_sha256='b'*64, raw_margin=.05, every_primary_pairs=50,
                epoch_local_counter=True, batch=32, mean_pool_weight=1., pair_weight_class_values=[2,1],
                tv_rng_seed=101 ^ 0xe3607476726e6700, normalization='lambda*w/(32*mean_pool_weight)',
                optimizer='pre-update gradient addition; one existing primary Adam boundary; no TV-only Adam', **{'lambda': 1.})
            with mock.patch.object(q.subprocess, 'run', side_effect=lambda *a, **k: SimpleNamespace(stdout=json.dumps(decoded))):
                self.assertEqual(q.verify_training(result, root/'checkpoint', cell.kind, expected, fixture.inspector), 'registered-fit')
                for mutation in ('row_mask', 'targets', 'weight', 'epoch', 'tv_lambda', 'input_hash', 'seed'):
                    bad = copy.deepcopy(result); prior = copy.deepcopy(decoded)
                    if mutation == 'row_mask': bad['selection']['e36_base_filter']['groups'][0]['stored_manifest']['row_selection_sha256']='0'*64
                    if mutation == 'targets': bad['selection']['e36_base_filter']['targets_changed']=1
                    if mutation == 'weight': decoded['repro']['inputs'][0]['train_w']=31.
                    if mutation == 'epoch': decoded['repro']['checkpoint_epoch']='118'
                    if mutation == 'tv_lambda': decoded['repro']['e36_tv']['lambda']=10.
                    if mutation == 'seed': decoded['repro']['sample_seed']=102
                    if mutation == 'input_hash': decoded['repro']['inputs'][0]['sha256']='0'*64
                    with self.assertRaises(ValueError, msg=mutation):
                        q.verify_training(bad, root/'checkpoint', cell.kind, expected, fixture.inspector)
                    decoded.clear(); decoded.update(prior)
            self.assertEqual(expected['input_contracts']['kadid']['human']['rows'], 2)

    def test_arm_specific_native_watchdog_and_no_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'benchmarks').mkdir()
            variants=[]
            for arm in e.ARMS:
                variants.append(dict(root='/base', spec='fixed', data_sha='d'*64, routes={'kadid': []},
                    schema='e36-research-fit-contract-v1',launchable=True,epochs=120, pairs_per_epoch=50000, selected_epoch=119, wall_cap_sec=86400,
                    e36=dict(arm=arm, contract_sha256='c'*64, base_filter_sha256='b'*64, trainer_sha256='a'*64)))
            (root/e.PACKAGE).write_text(json.dumps(dict(schema='e36-research-fit-package-v1', variants=variants)))
            (root/'build_meta.json').write_text(json.dumps(dict(files={'bin/zensim_mlp_train':'a'*64})))
            argv=['v2_lodo_mlp.py','--root','/base','--spec','fixed','--heldout','kadid',
                  '--e36-arm','tv1-c','--e36-tv-contract-sha256','c'*64,'--e36-base-filter-sha256','b'*64]
            self.assertEqual(fit_cell_exec.registered_wall_cap(argv,'d'*64,root),86400)
            package_path=root/e.PACKAGE
            for field,badvalue in [('launchable',False),('launchable',None),('schema','other')]:
                bad=copy.deepcopy(variants);bad[1][field]=badvalue
                package_path.write_text(json.dumps(dict(schema='e36-research-fit-package-v1',variants=bad)))
                with self.assertRaisesRegex(ValueError,'owner-blocked'):fit_cell_exec.registered_wall_cap(argv,'d'*64,root)
            package_path.write_text(json.dumps(dict(schema='e36-research-fit-package-v1',variants=variants)))
            policy=root/'bounded_smoke_policy.json';policy.write_text('{}')
            with self.assertRaisesRegex(ValueError,'mixed smoke'):fit_cell_exec.registered_wall_cap(argv,'d'*64,root)
            policy.unlink()

            for bad in (argv+['--local-smoke-budget','1:1'], [v if v!='tv1-c' else 'unknown' for v in argv],
                        [v if v!='b'*64 else '0'*64 for v in argv]):
                with self.assertRaises(ValueError): fit_cell_exec.registered_wall_cap(bad,'d'*64,root)

    def test_executor_cannot_export_unqualified_model_projection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);program=root/'program';program.mkdir();(program/'bin').mkdir()
            dest=root/'cell';dest.mkdir();(dest/'refit').mkdir()
            raw=dest/'refit/last.bin';raw.write_bytes(b'invented checkpoint')
            for p in [program/'bin/inspect_qualified_checkpoint',program/'qualified_fit_contract.py',program/'e36_fit_contract.py']:
                p.write_bytes(b'invented public owner')
            kind=dict(program_sha='a'*64,data_sha='b'*64,argv_sha='c'*64)
            binding=dict(raw_checkpoint_sha256=q.sha(raw),packed_model_sha256=None,program_sha256='a'*64,arm='tv0-c',selected_epoch=119)
            (dest/'e36-model-binding.json').write_text(json.dumps(binding))
            result=dict(selected_bake=str(raw),selected_bake_sha256=q.sha(raw))
            (dest/'result.json').write_text(json.dumps(result))
            expected=dict(route='kadid',e36=dict(arm='tv0-c'))
            with mock.patch.object(q,'trusted_contract',return_value=expected),mock.patch.object(q,'verify_training',side_effect=ValueError('decoded seed differs')):
                with self.assertRaisesRegex(ValueError,'decoded seed'):e.qualify_output(program,dest,kind,'cell',result)
            self.assertFalse((dest/'e36-qualified-checkpoint.json').exists())
            with mock.patch.object(q,'trusted_contract',return_value=expected),mock.patch.object(q,'verify_training',return_value='registered-fit'):
                e.qualify_output(program,dest,kind,'cell',result)
            receipt=json.loads((dest/'e36-qualified-checkpoint.json').read_text())
            self.assertEqual(receipt['raw_checkpoint_sha256'],q.sha(raw))
            self.assertEqual(receipt['model_binding_sha256'],q.sha(dest/'e36-model-binding.json'))
            self.assertEqual(receipt['result_sha256'],q.sha(dest/'result.json'))

    def test_e36_source_inventory_requires_native_import_closure(self):
        names=['v2_lodo_mlp.py','v2_confirm_fit.py','v2_common.py','v2_teacher.py','v2c_wide.py',
               'v2_human_role.py','v2_production_pack.py','rev5_bank.py','e33_fx1_declaration.json',
               'e36_training.py','e36_store.py']
        files={'scripts/rev4_featpot/'+n:'a'*64 for n in names}
        files.update({'scripts/lib/zen_stats.py':'a'*64,'scripts/lib/assessment_identity.py':'a'*64})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pins.json'
            record=dict(schema='e36-public-program-v1', files=files, data={e.PACKAGE:'b'*64})
            path.write_text(json.dumps(record))
            self.assertEqual(pack_fit_program.e36_sources(path,q.sha(path)),files)
            del files['scripts/rev4_featpot/e36_training.py']; path.write_text(json.dumps(record))
            with self.assertRaises(ValueError): pack_fit_program.e36_sources(path,q.sha(path))


if __name__ == '__main__': unittest.main()
