"""A swallowed device-service failure cannot qualify a returned GPU score."""
import unittest
from run import gpu_dispatch_failure
from parity import qualify_rows
class Dispatch(unittest.TestCase):
    def test_known_swallowed_panic_and_binding_rejection(self):
        self.assertTrue(gpu_dispatch_failure("thread 'DSD-0' panicked at stream.rs:122: binding alignment"))
        self.assertTrue(gpu_dispatch_failure('cubecl-wgpu: storage buffer binding offset 4 is not a multiple of 256'))
        self.assertTrue(gpu_dispatch_failure('wgpu error: Validation Error: too many workgroups'))
    def test_ordinary_runtime_messages_remain_eligible(self):
        self.assertFalse(gpu_dispatch_failure('cubecl-wgpu: adapter Apple M4 Pro'))
        self.assertFalse(gpu_dispatch_failure('IQA_ROW {"status":"ok"}'))
    def test_dispatch_failure_requests_an_independent_repeat(self):
        row=dict(status='skipped:gpu_dispatch_failure',score=9.26,n=10)
        self.assertTrue(qualify_rows([row],{},[],{}))
        self.assertEqual(row['status'],'skipped:gpu_dispatch_failure')
if __name__=='__main__':unittest.main()
