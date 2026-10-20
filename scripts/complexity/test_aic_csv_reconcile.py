import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('aic_csv_reconcile', Path(__file__).with_name('aic_csv_reconcile.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def table(**values):
    row = dict(source='S01_Ref_00.png', distorted='S01_A_01.png', **values)
    return {'columns': list(row), 'rows': {'S01_A_01.png': row}}


class Reconcile(unittest.TestCase):
    def test_changed_values_remain_visible_and_union_refuses(self):
        a, b = table(HDR_VDP_3='9.123456789012345'), table(HDR_VDP_3='9.223456789012345')
        r = m.compare(a, b)
        self.assertEqual(r['shared_keys'], 1)
        self.assertEqual(r['columns']['HDR_VDP_3']['numeric_changed'], 1)
        with self.assertRaises(ValueError):
            m.union_table(a, b, 'cropped', set())

    def test_alias_and_new_columns_keep_original_strings(self):
        a = table(PSNR='12.345678901234567', CKDN='0.23456789012345678')
        b = table(PSNR='12.345678901234567', **{'WS-HV': '0.98765432109876543'})
        columns, rows = m.union_table(a, b, 'cropped', {'S01_A_01.png'})
        self.assertIn('CKDN', columns)
        self.assertIn('WS-HV', columns)
        self.assertEqual(rows[0]['PSNR'], '12.345678901234567')
        self.assertEqual(rows[0]['ptc_subset'], '1')
        self.assertEqual(rows[0]['view'], 'cropped')
        b['rows']['S01_A_01.png']['distorted'] = 'PTC_S01_A_01.png'
        self.assertEqual(m.compare(a, b)['columns']['distorted']['equal'], 1)

    def test_representation_changes_are_not_metric_changes(self):
        r = m.compare(table(PSNR='1.000'), table(PSNR='1'))
        self.assertEqual(r['columns']['PSNR']['representation_only'], 1)
        self.assertEqual(r['columns']['PSNR']['numeric_changed'], 0)

    def test_missing_rows_are_counted(self):
        a, b = table(PSNR='1'), table(PSNR='1')
        b['rows']['S02_A_01.png'] = dict(source='S02_Ref_00.png', distorted='S02_A_01.png', PSNR='2')
        r = m.compare(a, b)
        self.assertEqual(r['right_only_keys'], ['S02_A_01.png'])


if __name__ == '__main__':
    unittest.main()
