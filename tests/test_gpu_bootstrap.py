import unittest
from unittest.mock import Mock, patch

from tools import ensure_gpu as boot


class BootstrapTests(unittest.TestCase):
    def result(self, code=0):
        return Mock(returncode=code, stdout='diagnostic')

    def test_explicit_cpu_never_installs(self):
        with patch.dict(boot.os.environ, CRBOT_COMPUTE_DEVICE='cpu'), patch.object(boot, 'run') as run:
            self.assertEqual(boot.ensure_gpu(), 0)
            run.assert_not_called()

    def test_no_nvidia_never_installs(self):
        with patch.dict(boot.os.environ, {}, clear=True), patch.object(boot, 'has_nvidia', return_value=False), patch.object(boot, 'run') as run:
            self.assertEqual(boot.ensure_gpu(), 0)
            run.assert_not_called()

    def test_cpu_wheel_is_replaced_and_kernel_rechecked(self):
        with patch.object(boot.Path, 'exists', return_value=True), patch.object(boot, 'run', side_effect=[self.result(1), self.result(), self.result(), self.result(), self.result()]) as run:
            self.assertEqual(boot.ensure_gpu(force=True), 0)
            install = run.call_args_list[1].args[0]
            self.assertIn('torch==2.11.0+cu128', install)
            self.assertIn('torchvision==0.26.0+cu128', install)
            self.assertEqual(run.call_args_list[-1].args[0][-1], boot.CUDA_CHECK)

    def test_fresh_clone_creates_environment_and_training_dependencies(self):
        with patch.object(boot.Path, 'exists', return_value=False), patch.object(boot, 'run', side_effect=[self.result(), self.result(1), self.result(), self.result(1), self.result(), self.result(), self.result()]) as run:
            self.assertEqual(boot.ensure_gpu(force=True), 0)
            self.assertIn('venv', run.call_args_list[0].args[0])
            self.assertIn('requirements-training.txt', run.call_args_list[4].args[0])

    def test_download_failure_stops_startup(self):
        with patch.object(boot.Path, 'exists', return_value=True), patch.object(boot, 'run', side_effect=[self.result(1), self.result(1)]) as run:
            self.assertEqual(boot.ensure_gpu(force=True), 1)
            self.assertEqual(run.call_count, 2)

    def test_working_cuda_not_reinstalled_and_failed_final_kernel_is_error(self):
        with patch.object(boot.Path, 'exists', return_value=True), patch.object(boot, 'run', side_effect=[self.result(), self.result(), self.result(), self.result(1)]) as run:
            self.assertEqual(boot.ensure_gpu(force=True), 1)
            self.assertFalse(any('install' in call.args[0] for call in run.call_args_list))


if __name__ == '__main__':
    unittest.main()
