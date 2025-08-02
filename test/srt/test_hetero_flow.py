import os
import time
import unittest
from itertools import product
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_HETEROFLOW,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestHeteroFlow(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_HETEROFLOW
        # cls.model = "/home/qually/local_ckpt/DeepSeek-V2-Lite-Chat-FP8/"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls.current_tp_size = None
        cls.current_tbo_heto = None
        cls.current_gpu_experts = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
            time.sleep(5)

    def setUp_server(self, tp_size: int, tbo_heto: bool, gpu_experts: int):
        """Setup server with specific tp_size"""
        if (
            self.process
            and self.current_tp_size == tp_size
            and self.current_tbo_heto == tbo_heto
            and self.current_gpu_experts == gpu_experts
        ):
            return  # Server already running with correct tp_size

        # Kill existing server if running
        if self.process:
            kill_process_tree(self.process.pid)
            time.sleep(5)

        other_args = [
            "--trust-remote-code",
            "--enable-ep-moe",
            "--tp",
            str(tp_size),
            "--max-running-requests",
            "32",
            "--attention-backend=triton",
            "--enable-ep-moe-heto",
            "--ep-moe-heto-gpu-experts",
            f"{gpu_experts}",
            "--cuda-graph-bs",
            "1",
            "2",
            "3",
            "4",
            "16",
            "32",
            "--disable-cuda-graph-padding",
            "--mem-fraction-static",
            "0.45",
        ]
        if tbo_heto:
            other_args = other_args + [
                "--enable-two-batch-overlap",
                "--two-batch-overlap-mode",
                "heto",
            ]

        # Start server with new tp_size
        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                "ENABLE_NUMA": "1",
                "OMP_NUM_THREADS": "16",
            }.update(
                os.environ
            ),  # Ensure to inherit existing environment variables
        )
        self.current_tp_size = tp_size
        self.current_tbo_heto = tbo_heto
        self.current_gpu_experts = gpu_experts
        self.config_name = (
            f"tp{tp_size}_{'tbo_heto_' if tbo_heto else ''}gpu_experts_{gpu_experts}"
        )

    def run_mgsm_en_test(self):
        """Run MGSM EN test with current server configuration"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )
        metrics = run_eval(args)
        assert metrics["score"] >= 0.7

    def run_gsm8k_test(self):
        """Run GSM8K test with current server configuration"""
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=100,
            parallel=64,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Eval accuracy of GSM8K with {self.config_name}: {metrics=}")
        self.assertGreater(metrics["accuracy"], 0.63)

    def test_gsm8k(self):
        tp_sizes = [1]
        tbo_heto_options = [False, True]
        gpu_experts_options = [0, 1, 63, 64]
        for tp_size, tbo_heto, gpu_experts in product(
            tp_sizes, tbo_heto_options, gpu_experts_options
        ):
            with self.subTest(
                tp_size=tp_size, tbo_heto=tbo_heto, gpu_experts=gpu_experts
            ):
                self.setUp_server(
                    tp_size=tp_size, tbo_heto=tbo_heto, gpu_experts=gpu_experts
                )
                self.run_gsm8k_test()


if __name__ == "__main__":
    unittest.main()
