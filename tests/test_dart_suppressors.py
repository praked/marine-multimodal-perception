"""apply_suppressors (DART prompt-v1): suppressor pseudo-detections kill
overlapping target-class boxes by intersection-over-smaller-area and are
never emitted themselves."""
import yaml

from scripts.gpu_corpus.label_bundle_dart import DEFAULT_CONFIG, apply_suppressors


def det(cls, xyxy, score=0.9, **kw):
    return {"cls": cls, "prompt": cls, "xyxy": list(xyxy), "score": score, **kw}


def sup(xyxy, targets, score=0.5):
    return {"cls": "_suppressor", "prompt": "outboard motor",
            "xyxy": list(xyxy), "score": score, "targets": targets}


def test_no_suppressors_is_passthrough():
    dets = [det("person", (0.1, 0.1, 0.2, 0.3)), det("boat", (0.4, 0.4, 0.6, 0.5))]
    assert apply_suppressors(dets) == dets


def test_small_motor_box_kills_containing_person_box():
    # motor box fully inside the person FP: IoU ~0.25 but IoS = 1.0
    person = det("person", (0.10, 0.10, 0.30, 0.50))
    motor = sup((0.15, 0.30, 0.25, 0.48), ["person"])
    out = apply_suppressors([person, motor])
    assert out == []


def test_non_target_class_untouched():
    boat = det("boat", (0.10, 0.10, 0.30, 0.50))
    motor = sup((0.15, 0.30, 0.25, 0.48), ["person"])
    assert apply_suppressors([boat, motor]) == [boat]


def test_ios_threshold_respected():
    person = det("person", (0.10, 0.10, 0.30, 0.50))
    barely = sup((0.28, 0.45, 0.40, 0.60), ["person"])  # tiny corner overlap
    assert apply_suppressors([person, barely], ios_thr=0.5) == [person]


def test_suppressor_never_emitted_even_alone():
    assert apply_suppressors([sup((0.1, 0.1, 0.2, 0.2), ["person"])]) == []


def test_config_is_v0_with_suppressors_rejected():
    # v1/v1.1 suppressor prompts were tested and REJECTED on the tranche-A
    # audit (2026-09-02): any added prompt perturbs SAM3's joint scoring.
    # The shipped config must stay v0 with NO suppressors; the mechanism
    # remains available (tests above) for future, tranche-validated use.
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert cfg["prompt_set"] == "v0"
    assert not cfg.get("suppressors")
