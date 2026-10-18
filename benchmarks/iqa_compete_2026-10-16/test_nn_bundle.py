"""Real ONNX external-tensor discovery controls; run with the preparation interpreter."""
from pathlib import Path
import os
import tempfile
import unittest
import onnx
from onnx import TensorProto, helper
from prepare_nn_durable import model_files


class ModelFilesTests(unittest.TestCase):
    def test_constant_tensor_external_file_is_required_and_contained(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp)
            (root / 'fixture.meta').write_text('channels 1\nmin_size 1\n')
            tensor = TensorProto(name='weight', data_type=TensorProto.FLOAT, dims=[1],
                                 data_location=TensorProto.EXTERNAL)
            tensor.external_data.add(key='location', value='weights-without-extension')
            node = helper.make_node('Constant', [], ['y'], value=tensor)
            graph = helper.make_graph([node], 'external', [], [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])])
            model = helper.make_model(graph)
            path = root / 'fixture_full.onnx'
            path.write_bytes(model.SerializeToString())
            with self.assertRaises(FileNotFoundError):
                model_files(root, 'fixture')
            weights = root / 'weights-without-extension'
            weights.write_bytes(b'\x00\x00\x80\x3f')
            self.assertEqual(model_files(root, 'fixture'), {path, weights, root / 'fixture.meta'})
            # Absolute external references must not escape the supplied export directory.
            model.graph.node[0].attribute[0].t.external_data[0].value = str(Path(__file__).resolve())
            path.write_bytes(model.SerializeToString())
            with self.assertRaisesRegex(ValueError, 'escapes'):
                model_files(root, 'fixture')


if __name__ == '__main__':
    unittest.main()
