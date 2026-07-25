"""
tests/test_env_smoke.py
End-to-end smoke test: verifies that the entire toolchain (PyTorch with CUDA,
timm EfficientNet-B4, and sentence-transformers) is correctly installed and
mutually compatible.

Run with: pytest tests/test_env_smoke.py -v
All 4 tests must pass before Phase 0 is considered complete.
"""
import torch


def test_cuda_available():
    """CUDA must be available — CPU training is not viable for this project."""
    assert torch.cuda.is_available(), "CUDA must be available for this project"


def test_basic_tensor_op_on_gpu():
    """Verify a basic forward pass through a Conv2d layer runs on the GPU."""
    x = torch.randn(4, 3, 224, 224, device="cuda")
    y = torch.nn.Conv2d(3, 16, 3).cuda()(x)
    assert y.shape == (4, 16, 222, 222)


def test_timm_model_loads():
    """Verify timm can instantiate EfficientNet-B4 (feature extractor mode)."""
    import timm

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
    assert model is not None


def test_sentence_transformer_loads():
    """Verify sentence-transformers can load and encode with all-MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(["stop", "go up"])
    assert emb.shape[0] == 2
