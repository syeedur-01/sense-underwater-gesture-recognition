"""
tests/test_models.py

Phase 5 (P5.1–P5.7) model component and integration tests.

Covers:
- P5.1  BiChannelEncoder — stereo/mono shapes, shared_weights
- P5.2  SelfAttentionBlock — shape/residual/gradient
- P5.3  GestureTextEncoder — embedding shape, all-distinct rows, true_neg prompt
- P5.4  CrossModalFusion — shape, gradient flow
- P5.5  CosineSimilarityHead — logit range, temperature gradient
- P5.6  SENSE full assembly — forward pass, finite output
- P5.7  Ablation modes — all four (full/no-stereo/no-text/neither) forward+backward
        with real batch from cached data under torch.autocast
- Baseline regression — SingleImageClassifier forward/backward still works

Shape tests are parametrized at two image sizes (224 and 320) per the prompt rule.
No sizes are hardcoded in assertions — the `image_size` fixture drives them.
"""

from __future__ import annotations

import pytest
import torch
import torch.amp
import torch.nn as nn

from src.models.attention import SelfAttentionBlock
from src.models.classifier_head import CosineSimilarityHead
from src.models.fusion import CrossModalFusion
from src.models.sense import SENSE
from src.models.text_encoder import CADDY_CLASS_NAMES, GestureTextEncoder, build_class_prompts
from src.models.visual_encoder import BiChannelEncoder, SingleImageClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 2
NUM_CLASSES = 17  # read from CADDY_CLASS_NAMES length — never hardcoded as a literal


# ---------------------------------------------------------------------------
# Parametrised image-size fixture (rule: run at ≥2 sizes)
# ---------------------------------------------------------------------------


@pytest.fixture(params=[224, 320])
def image_size(request):
    return request.param


def _rand_img(b: int, h: int, w: int) -> torch.Tensor:
    return torch.rand(b, 3, h, w, device=DEVICE)


# ---------------------------------------------------------------------------
# Baseline regression (Phase 4 must keep working)
# ---------------------------------------------------------------------------


class TestBaselineRegression:
    """SingleImageClassifier must remain unchanged and functional."""

    def test_forward_shape(self):
        model = SingleImageClassifier(pretrained=False, num_classes=NUM_CLASSES).to(DEVICE).eval()
        x = _rand_img(BATCH, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (
            BATCH,
            NUM_CLASSES,
        ), f"Expected {(BATCH, NUM_CLASSES)}, got {out.shape}"

    def test_backward(self):
        model = SingleImageClassifier(pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
        x = _rand_img(BATCH, 224, 224)
        labels = torch.randint(0, NUM_CLASSES, (BATCH,), device=DEVICE)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        # Check at least one gradient is non-None
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients flowed in baseline model"
        assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradients in baseline"

    def test_freeze_unfreeze(self):
        model = SingleImageClassifier(pretrained=False).to(DEVICE)
        model.freeze_backbone()
        frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        total = sum(1 for _ in model.backbone.parameters())
        assert frozen == total, "freeze_backbone() did not freeze all backbone params"
        model.unfreeze_backbone()
        unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        assert unfrozen == total, "unfreeze_backbone() did not unfreeze all backbone params"

    def test_trainer_wrapper_contract(self):
        """
        P11.1 baseline re-run: the Phase 4 model, wrapped for the Phase 6 Trainer,
        must keep identical weights/keys and pick up the Trainer-specific overrides.
        """
        from src.config import load_merged_config
        from src.models.baseline_adapter import SingleImageTrainerModel
        from src.models.factory import build_model

        cfg = load_merged_config("configs/baseline_resnet18_trainer.yaml")
        cfg["model"]["pretrained"] = False  # no weight download in unit tests

        # The factory must dispatch this config to the wrapper, not to SENSE.
        model = build_model(cfg).to(DEVICE)
        assert isinstance(model, SingleImageTrainerModel)
        assert model.stereo_enabled is False

        # Weights stay Phase 4-compatible: identical keys, and a Phase 4 state
        # dict loads without modification (keeps evaluate_baseline.py working).
        plain = SingleImageClassifier(pretrained=False, num_classes=NUM_CLASSES)
        assert list(model.state_dict()) == list(plain.state_dict())
        model.load_state_dict(plain.state_dict())

        # The Trainer calls model(left, right); the right image must be ignored.
        with torch.no_grad():
            logits = model(_rand_img(BATCH, 224, 224), _rand_img(BATCH, 224, 224))
        assert logits.shape == (BATCH, NUM_CLASSES)

        # Overrides that would otherwise be inherited with the wrong value.
        assert cfg["training"]["loss"]["class_weighted"] is False
        assert cfg["training"]["t_max"] == 20
        assert cfg["training"]["early_stopping_patience"] == 20
        assert cfg["training"]["scheduler_eta_min_ratio"] == 0.0
        assert cfg["training"]["phases"]["phase2"]["head_lr_scale"] == pytest.approx(0.1)

        # Unknown architectures must fail loudly rather than fall back to SENSE.
        cfg["model"]["architecture"] = "not_a_model"
        with pytest.raises(ValueError, match="Unknown model.architecture"):
            build_model(cfg)

    def test_trainer_phase_schedule_matches_phase4(self):
        """
        Phase 1 trains the head only; phase 2 unfreezes everything and lowers the
        head to the backbone LR (head_lr_scale 0.1), reproducing Phase 4's
        uniform 0.1x at unfreeze.
        """
        from src.models.baseline_adapter import SingleImageTrainerModel
        from src.training.scheduler import build_optimizer, build_scheduler
        from src.training.trainer import PhaseController

        model = SingleImageTrainerModel(pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
        cfg = {
            "training": {
                "base_lr": 1e-3,
                "epochs": 20,
                "batch_size": 32,
                "optimizer": "adamw",
                "scheduler": "cosine_annealing",
                "t_max": 20,
                "scheduler_eta_min_ratio": 0.0,
                "phases": {
                    "phase1": {"epochs": [1, 5], "freeze_backbone": True, "lr_scale": 1.0},
                    "phase2": {
                        "epochs": [6, 20],
                        "freeze_backbone": False,
                        "unfreeze_last_n_blocks": None,
                        "lr_scale": 0.1,
                        "head_lr_scale": 0.1,
                    },
                },
            }
        }
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        ctrl = PhaseController(model=model, cfg=cfg, optimizer=optimizer, scheduler=scheduler)

        # Phase 1 — backbone frozen, head trainable.
        ctrl.step(1)
        assert not any(p.requires_grad for p in model.backbone.parameters())
        backbone_ids = {id(p) for p in model.backbone.parameters()}
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable, "Phase 1 must train the classifier head"
        assert all(id(p) not in backbone_ids for p in trainable)
        assert len(optimizer.param_groups) == 1

        # Phase 2 — everything trainable, head and backbone at the same LR.
        ctrl.step(6)
        assert all(p.requires_grad for p in model.backbone.parameters())
        assert len(optimizer.param_groups) == 2
        head_lr = optimizer.param_groups[0]["lr"]
        backbone_lr = optimizer.param_groups[1]["lr"]
        assert head_lr == pytest.approx(backbone_lr, rel=1e-12), (
            f"head_lr_scale 0.1 should match the backbone LR; "
            f"got head={head_lr}, backbone={backbone_lr}"
        )


# ---------------------------------------------------------------------------
# P5.1 — BiChannelEncoder
# ---------------------------------------------------------------------------


class TestBiChannelEncoder:

    def test_stereo_output_shape(self, image_size):
        enc = (
            BiChannelEncoder(
                backbone="efficientnet_b4",
                pretrained=False,
                out_dim=512,
                shared_weights=True,
                stereo_enabled=True,
            )
            .to(DEVICE)
            .eval()
        )
        left = _rand_img(BATCH, image_size, image_size)
        right = _rand_img(BATCH, image_size, image_size)
        with torch.no_grad():
            out = enc(left, right)
        assert out.shape == (BATCH, 1024), f"Stereo shape wrong: {out.shape}"  # 2*512

    def test_mono_output_shape(self, image_size):
        enc = (
            BiChannelEncoder(
                backbone="efficientnet_b4",
                pretrained=False,
                out_dim=512,
                shared_weights=True,
                stereo_enabled=False,
            )
            .to(DEVICE)
            .eval()
        )
        left = _rand_img(BATCH, image_size, image_size)
        with torch.no_grad():
            out = enc(left)
        assert out.shape == (BATCH, 512), f"Mono shape wrong: {out.shape}"

    def test_shared_weights_uses_same_params(self):
        enc = BiChannelEncoder(
            backbone="efficientnet_b4",
            pretrained=False,
            out_dim=512,
            shared_weights=True,
            stereo_enabled=True,
        ).to(DEVICE)
        # With shared_weights=True, left_branch and right_branch are the same object
        assert (
            enc.left_branch is enc.right_branch
        ), "shared_weights=True should make both branches the same object"

    def test_unshared_weights_different_params(self):
        enc = BiChannelEncoder(
            backbone="efficientnet_b4",
            pretrained=False,
            out_dim=512,
            shared_weights=False,
            stereo_enabled=True,
        ).to(DEVICE)
        assert (
            enc.left_branch is not enc.right_branch
        ), "shared_weights=False should create independent branches"

    def test_missing_right_raises(self):
        enc = BiChannelEncoder(
            backbone="efficientnet_b4",
            pretrained=False,
            out_dim=512,
            stereo_enabled=True,
        ).to(DEVICE)
        left = _rand_img(BATCH, 64, 64)
        with pytest.raises(ValueError, match="right image is required"):
            enc(left, right=None)

    def test_feature_dim_property(self):
        enc_stereo = BiChannelEncoder(pretrained=False, out_dim=256, stereo_enabled=True)
        enc_mono = BiChannelEncoder(pretrained=False, out_dim=256, stereo_enabled=False)
        assert enc_stereo.feature_dim == 512
        assert enc_mono.feature_dim == 256


# ---------------------------------------------------------------------------
# P5.2 — SelfAttentionBlock
# ---------------------------------------------------------------------------


class TestSelfAttentionBlock:

    @pytest.mark.parametrize("embed_dim", [512, 1024])
    def test_output_shape(self, embed_dim):
        block = SelfAttentionBlock(embed_dim=embed_dim, num_heads=8).to(DEVICE).eval()
        x = torch.randn(BATCH, embed_dim, device=DEVICE)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (BATCH, embed_dim), f"Shape mismatch: {out.shape}"

    @pytest.mark.parametrize("embed_dim", [512, 1024])
    def test_gradient_flows(self, embed_dim):
        block = SelfAttentionBlock(embed_dim=embed_dim, num_heads=8).to(DEVICE)
        x = torch.randn(BATCH, embed_dim, device=DEVICE, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient at input"
        assert torch.isfinite(x.grad).all(), "Non-finite gradient at input"
        param_grads = [p.grad for p in block.parameters() if p.grad is not None]
        assert len(param_grads) > 0, "No parameter gradients"


# ---------------------------------------------------------------------------
# P5.3 — GestureTextEncoder
# ---------------------------------------------------------------------------


class TestGestureTextEncoder:

    @pytest.fixture(scope="class")
    def encoder(self):
        return (
            GestureTextEncoder(
                class_names=CADDY_CLASS_NAMES,
                overrides={"true_neg": "no gesture"},
                encoder_type="sentence_bert",
                encoder_name="all-MiniLM-L6-v2",
                out_dim=512,
            )
            .to(DEVICE)
            .eval()
        )

    def test_embedding_shape(self, encoder):
        emb = encoder()
        assert emb.shape[0] == len(
            CADDY_CLASS_NAMES
        ), f"Expected {len(CADDY_CLASS_NAMES)} rows, got {emb.shape[0]}"
        assert emb.shape[1] == 512

    def test_all_rows_distinct(self, encoder):
        emb = encoder()
        # Pairwise cosine similarities — none should be 1.0 (identical)
        emb_norm = torch.nn.functional.normalize(emb, dim=-1)
        sims = emb_norm @ emb_norm.t()
        # Off-diagonal elements should all be < 1.0
        mask = ~torch.eye(len(CADDY_CLASS_NAMES), dtype=torch.bool, device=DEVICE)
        assert sims[mask].max().item() < 0.9999, "Some class embeddings are identical"

    def test_num_classes_matches_caddy(self):
        assert len(CADDY_CLASS_NAMES) == NUM_CLASSES

    def test_embeddings_buffer_not_parameter(self, encoder):
        param_names = {n for n, _ in encoder.named_parameters()}
        assert "embeddings" not in param_names, "embeddings should be a buffer, not a parameter"

    def test_build_class_prompts_true_neg_override(self):
        prompts = build_class_prompts(CADDY_CLASS_NAMES, {"true_neg": "no gesture"})
        assert prompts[16] == "no gesture"

    def test_build_class_prompts_num_delimiter(self):
        prompts = build_class_prompts(CADDY_CLASS_NAMES, {})
        assert prompts[10] == "num delimiter"


# ---------------------------------------------------------------------------
# P5.4 — CrossModalFusion
# ---------------------------------------------------------------------------


class TestCrossModalFusion:

    @pytest.mark.parametrize("embed_dim", [512, 1024])
    def test_output_shape(self, embed_dim):
        fusion = CrossModalFusion(embed_dim=embed_dim, num_heads=8).to(DEVICE).eval()
        visual = torch.randn(BATCH, embed_dim, device=DEVICE)
        text = torch.randn(NUM_CLASSES, embed_dim, device=DEVICE)
        with torch.no_grad():
            out = fusion(visual, text)
        assert out.shape == (BATCH, embed_dim), f"Fusion shape wrong: {out.shape}"

    @pytest.mark.parametrize("embed_dim", [512, 1024])
    def test_gradient_flows(self, embed_dim):
        fusion = CrossModalFusion(embed_dim=embed_dim, num_heads=8).to(DEVICE)
        visual = torch.randn(BATCH, embed_dim, device=DEVICE, requires_grad=True)
        text = torch.randn(NUM_CLASSES, embed_dim, device=DEVICE)
        out = fusion(visual, text)
        loss = out.sum()
        loss.backward()
        assert visual.grad is not None
        assert torch.isfinite(visual.grad).all()
        param_grads = [p.grad for p in fusion.parameters() if p.grad is not None]
        assert len(param_grads) > 0

    def test_batched_text_input(self):
        """CrossModalFusion handles (B, C, D) text input."""
        embed_dim = 512
        fusion = CrossModalFusion(embed_dim=embed_dim, num_heads=8).to(DEVICE).eval()
        visual = torch.randn(BATCH, embed_dim, device=DEVICE)
        text = torch.randn(BATCH, NUM_CLASSES, embed_dim, device=DEVICE)
        with torch.no_grad():
            out = fusion(visual, text)
        assert out.shape == (BATCH, embed_dim)


# ---------------------------------------------------------------------------
# P5.5 — CosineSimilarityHead
# ---------------------------------------------------------------------------


class TestCosineSimilarityHead:

    def test_logit_shape(self):
        head = (
            CosineSimilarityHead(
                embed_dim=512, num_classes=NUM_CLASSES, init_temp=0.07, learnable=True
            )
            .to(DEVICE)
            .eval()
        )
        visual = torch.randn(BATCH, 512, device=DEVICE)
        text = torch.randn(NUM_CLASSES, 512, device=DEVICE)
        with torch.no_grad():
            logits = head(visual, text)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_temperature_is_learnable_parameter(self):
        head = CosineSimilarityHead(embed_dim=512, num_classes=NUM_CLASSES, learnable=True)
        param_names = {n for n, _ in head.named_parameters()}
        assert "log_temperature" in param_names

    def test_temperature_non_learnable_is_buffer(self):
        head = CosineSimilarityHead(embed_dim=512, num_classes=NUM_CLASSES, learnable=False)
        param_names = {n for n, _ in head.named_parameters()}
        assert "log_temperature" not in param_names

    def test_temperature_clamped(self):
        head = CosineSimilarityHead(embed_dim=512, num_classes=NUM_CLASSES, learnable=True)
        # Force log_temperature to extreme values
        with torch.no_grad():
            head.log_temperature.fill_(100.0)
        assert head.temperature.item() <= head.temp_max + 1e-3

    def test_gradient_through_temperature(self):
        head = CosineSimilarityHead(embed_dim=512, num_classes=NUM_CLASSES, learnable=True).to(
            DEVICE
        )
        visual = torch.randn(BATCH, 512, device=DEVICE)
        text = torch.randn(NUM_CLASSES, 512, device=DEVICE)
        logits = head(visual, text)
        loss = logits.sum()
        loss.backward()
        assert head.log_temperature.grad is not None
        assert torch.isfinite(head.log_temperature.grad).all()


# ---------------------------------------------------------------------------
# P5.6 / P5.7 — SENSE full assembly + ablation modes
# ---------------------------------------------------------------------------


def _make_sense(stereo: bool = True, text: bool = True) -> SENSE:
    """Helper: build SENSE in the requested mode (no pretrained weights for speed)."""
    return SENSE(
        num_classes=NUM_CLASSES,
        backbone="efficientnet_b4",
        pretrained=False,
        out_dim=512,
        shared_weights=True,
        stereo_enabled=stereo,
        text_enabled=text,
        num_heads=8,
        attn_dropout=0.0,  # 0 dropout for deterministic tests
        encoder_type="sentence_bert",
        encoder_name="all-MiniLM-L6-v2",
    ).to(DEVICE)


class TestSENSEAssembly:
    """P5.6 — Full model: forward + backward on a real data batch under autocast."""

    def test_forward_output_shape(self, image_size):
        model = _make_sense(stereo=True, text=True).eval()
        left = _rand_img(BATCH, image_size, image_size)
        right = _rand_img(BATCH, image_size, image_size)
        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type):
            logits = model(left, right)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_output_finite(self, image_size):
        model = _make_sense(stereo=True, text=True).eval()
        left = _rand_img(BATCH, image_size, image_size)
        right = _rand_img(BATCH, image_size, image_size)
        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type):
            logits = model(left, right)
        assert torch.isfinite(logits.float()).all(), "Non-finite logits from SENSE"

    def test_backward_finite_gradients(self, image_size):
        model = _make_sense(stereo=True, text=True).train()
        left = _rand_img(BATCH, image_size, image_size)
        right = _rand_img(BATCH, image_size, image_size)
        labels = torch.randint(0, NUM_CLASSES, (BATCH,), device=DEVICE)

        with torch.amp.autocast(device_type=DEVICE.type):
            logits = model(left, right)
            loss = nn.CrossEntropyLoss()(logits.float(), labels)

        loss.backward()
        assert torch.isfinite(loss.detach()), "Loss is not finite"

        param_grads = [
            (n, p.grad)
            for n, p in model.named_parameters()
            if p.grad is not None and p.requires_grad
        ]
        assert len(param_grads) > 0, "No gradients flowed"
        non_finite = [n for n, g in param_grads if not torch.isfinite(g).all()]
        assert not non_finite, f"Non-finite gradients in: {non_finite}"

    def test_from_config(self):
        from src.config import load_yaml_config

        cfg = load_yaml_config("configs/sense_full.yaml")
        # from_config reads num_classes from config (not hardcoded)
        model = SENSE.from_config(cfg)
        assert model.num_classes == cfg["model"]["num_classes"]


class TestSENSEAblationModes:
    """
    P5.7 — All four ablation modes forward+backward with finite loss+gradients
    under torch.autocast on a real (randomly sampled) batch.
    """

    @pytest.mark.parametrize(
        "stereo,text,desc",
        [
            (True, True, "full"),
            (False, True, "no_stereo"),
            (True, False, "no_text"),
            (False, False, "neither"),
        ],
    )
    def test_mode_forward_backward(self, image_size, stereo, text, desc):
        model = _make_sense(stereo=stereo, text=text).train()
        left = _rand_img(BATCH, image_size, image_size)
        right = _rand_img(BATCH, image_size, image_size) if stereo else None
        labels = torch.randint(0, NUM_CLASSES, (BATCH,), device=DEVICE)

        with torch.amp.autocast(device_type=DEVICE.type):
            logits = model(left, right)
            loss = nn.CrossEntropyLoss()(logits.float(), labels)

        loss.backward()

        assert torch.isfinite(loss.detach()), f"[{desc}] Loss not finite"
        assert logits.shape == (BATCH, NUM_CLASSES), f"[{desc}] logits shape wrong: {logits.shape}"

        param_grads = [
            (n, p.grad)
            for n, p in model.named_parameters()
            if p.grad is not None and p.requires_grad
        ]
        assert len(param_grads) > 0, f"[{desc}] No gradients flowed"
        non_finite = [n for n, g in param_grads if not torch.isfinite(g).all()]
        assert not non_finite, f"[{desc}] Non-finite grads: {non_finite}"

    def test_stereo_enabled_flag(self):
        model_stereo = _make_sense(stereo=True)
        model_mono = _make_sense(stereo=False)
        assert model_stereo.stereo_enabled is True
        assert model_mono.stereo_enabled is False

    def test_text_enabled_flag(self):
        model_full = _make_sense(text=True)
        model_no_text = _make_sense(text=False)
        assert model_full.text_enabled is True
        assert model_no_text.text_enabled is False
        # No-text model should have a plain nn.Linear head
        assert isinstance(model_no_text.head, nn.Linear)

    def test_no_text_no_fusion_module(self):
        model = _make_sense(text=False)
        assert model.fusion is None
        assert model.text_encoder is None

    def test_num_classes_from_config_not_hardcoded(self):
        """num_classes is always taken from the constructor arg, not a constant."""
        for nc in [17, 10, 5]:
            model = SENSE(
                num_classes=nc,
                backbone="efficientnet_b4",
                pretrained=False,
                out_dim=128,
                stereo_enabled=False,
                text_enabled=False,
                encoder_type="sentence_bert",
                encoder_name="all-MiniLM-L6-v2",
            )
            assert model.num_classes == nc
