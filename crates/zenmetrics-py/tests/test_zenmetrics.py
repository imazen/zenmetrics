"""Unit tests for zenmetrics Python package."""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import zenmetrics as zm

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "crates" / "zenmetrics-cli" / "tests" / "fixtures"
REF_64 = str(FIXTURES_DIR / "ref_64.png")
DIST_ID_64 = str(FIXTURES_DIR / "dist_identical_64.png")
DIST_NOISY_64 = str(FIXTURES_DIR / "dist_noisy_64.png")


def test_list_metrics():
    metrics = zm.list_metrics()
    assert isinstance(metrics, list)
    assert len(metrics) > 30
    assert "ssim2" in metrics
    assert "butteraugli" in metrics
    assert "psnr" in metrics
    assert "cvvdp" in metrics


def test_score_files():
    res = zm.score(REF_64, DIST_ID_64, metric="ssim2")
    assert isinstance(res, zm.ScoreResult)
    assert float(res) == 100.0
    assert res.value == 100.0
    assert "ssim2" in res.scores

    res_noisy = zm.score(REF_64, DIST_NOISY_64, metric="ssim2")
    assert float(res_noisy) < 50.0
    assert res_noisy < res


def test_score_operators():
    res = zm.score(REF_64, DIST_ID_64, metric="ssim2")
    assert res > 99.0
    assert res >= 100.0
    assert res == 100.0
    assert int(res) == 100
    assert float(res) == 100.0
    assert f"{res:.2f}" == "100.00"
    assert (res + 10.0) == 110.0
    assert (10.0 + res) == 110.0
    assert (res - 10.0) == 90.0
    assert (res * 2.0) == 200.0
    assert (res / 2.0) == 50.0
    assert "ssim2" in res
    assert res["ssim2"] == 100.0
    assert res.get("ssim2") == 100.0
    assert res.get("nonexistent") is None


def test_score_numpy_uint8():
    arr_ref = np.zeros((64, 64, 3), dtype=np.uint8)
    arr_dist = np.full((64, 64, 3), 10, dtype=np.uint8)

    res = zm.score(arr_ref, arr_dist, metric="psnr")
    assert isinstance(res, zm.ScoreResult)
    assert 28.0 < res.value < 29.0


def test_score_numpy_float_and_chw():
    rng = np.random.RandomState(42)
    # HWC float in [0, 1]
    hwc = rng.rand(64, 64, 3).astype(np.float32)
    hwc_dist = np.clip(hwc + 0.05, 0.0, 1.0)
    res_hwc = zm.score(hwc, hwc_dist, metric="psnr")
    assert res_hwc.value > 20.0

    # CHW float
    chw = np.transpose(hwc, (2, 0, 1))
    chw_dist = np.transpose(hwc_dist, (2, 0, 1))
    res_chw = zm.score(chw, chw_dist, metric="psnr")
    assert abs(res_chw.value - res_hwc.value) < 1e-4

    # Grayscale (H, W)
    gray = hwc[:, :, 0]
    gray_dist = hwc_dist[:, :, 0]
    res_gray = zm.score(gray, gray_dist, metric="psnr")
    assert res_gray.value > 20.0


def test_score_pil():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr_d = np.full((64, 64, 3), 10, dtype=np.uint8)
    p_ref = Image.fromarray(arr)
    p_dist = Image.fromarray(arr_d)

    res = zm.score(p_ref, p_dist, metric="psnr")
    assert 28.0 < res.value < 29.0


def test_callable_metric():
    m = zm.create_metric("butteraugli")
    assert isinstance(m, zm.Metric)
    assert m.name == "butteraugli"
    assert not m.higher_better

    res = m(REF_64, DIST_ID_64)
    assert res.value == 0.0


def test_compare():
    comp = zm.compare(
        REF_64,
        {
            "identical": DIST_ID_64,
            "noisy": DIST_NOISY_64,
        },
        metrics=["ssim2", "psnr"],
    )
    assert len(comp) == 2
    rows = comp.to_dict()
    assert rows[0]["variant"] == "identical"
    assert rows[0]["ssim2"] == 100.0
    assert rows[1]["variant"] == "noisy"
    assert rows[1]["ssim2"] < 50.0

    # Test array compare
    arr_ref = np.zeros((64, 64, 3), dtype=np.uint8)
    comp_arr = zm.compare(
        arr_ref,
        {"zero": arr_ref, "ten": np.full((64, 64, 3), 10, dtype=np.uint8)},
        metrics=["psnr"],
    )
    assert len(comp_arr) == 2


def test_score_folder():
    ref_dir = REPO_ROOT / "crates" / "cvvdp-conformance" / "data" / "hdr10_sky_192x108" / "ref"
    dist_dir = REPO_ROOT / "crates" / "cvvdp-conformance" / "data" / "hdr10_sky_192x108" / "dist_q8"

    res = zm.score_folder(ref_dir, dist_dir, metric="ssim", recursive=False)
    assert res.total_pairs == 24
    assert res.successful_pairs == 24
    assert res.failed_pairs == 0
    assert "ssim" in res.summary
    assert res.summary["ssim"]["count"] == 24
    assert len(res.results) == 24


def test_stats_panel():
    preds = [1.0, 2.0, 3.0, 4.0, 5.0]
    targets = [1.1, 1.9, 3.2, 3.8, 5.1]

    assert zm.stats.spearman(preds, targets) == 1.0
    assert zm.stats.pearson(preds, targets) > 0.99
    assert zm.stats.kendall_tau(preds, targets) == 1.0
    assert zm.stats.outlier_ratio(preds, targets) == 0.0

    panel = zm.stats.compute_panel(preds, targets)
    assert isinstance(panel, zm.stats.PanelStatsResult)
    assert panel.srocc == 1.0
    assert panel.plcc > 0.99
    assert panel.n == 5


def test_html_renderers():
    res = zm.score(REF_64, DIST_ID_64, metric="ssim2")
    html1 = res._repr_html_()
    assert "zm-card" in html1
    assert "SSIM2" in html1
    assert "100.00" in html1

    comp = zm.compare(REF_64, [DIST_ID_64, DIST_NOISY_64], metrics=["psnr"])
    html2 = comp._repr_html_()
    assert "zm-table" in html2
    assert "psnr" in html2

    panel = zm.stats.compute_panel([1.0, 2.0], [1.0, 2.0])
    html3 = panel._repr_html_()
    assert "SROCC" in html3
    assert "PLCC" in html3


def test_dataframes_and_polars():
    comp = zm.compare(
        REF_64,
        {"identical": DIST_ID_64, "noisy": DIST_NOISY_64},
        metrics=["ssim2", "psnr"],
    )
    # Pandas DataFrame
    df_pd = comp.to_dataframe()
    assert hasattr(df_pd, "shape")
    assert df_pd.shape == (2, 4)

    # Polars DataFrame
    df_pl = comp.to_polars()
    assert hasattr(df_pl, "shape")
    assert df_pl.shape == (2, 4)

    # Folder DataFrame
    ref_dir = REPO_ROOT / "crates" / "cvvdp-conformance" / "data" / "hdr10_sky_192x108" / "ref"
    dist_dir = REPO_ROOT / "crates" / "cvvdp-conformance" / "data" / "hdr10_sky_192x108" / "dist_q8"
    folder_res = zm.score_folder(ref_dir, dist_dir, metric="ssim", recursive=False)

    df_folder_pd = folder_res.to_dataframe()
    assert len(df_folder_pd) == 24
    df_folder_pl = folder_res.to_polars()
    assert len(df_folder_pl) == 24


def test_cvvdp_display_requirement():
    """Verify CVVDP strictly requires display_model to maintain scientific calibration integrity."""
    arr_ref = np.zeros((64, 64, 3), dtype=np.uint8)
    arr_dist = np.full((64, 64, 3), 10, dtype=np.uint8)

    # Omitting display_model must fail with an informative ValueError
    with pytest.raises(ValueError, match="requires an explicit display model"):
        zm.score(arr_ref, arr_dist, metric="cvvdp")

    # Providing valid display_model succeeds
    res = zm.score(arr_ref, arr_dist, metric="cvvdp", display_model="standard_4k")
    assert isinstance(res, zm.ScoreResult)
    assert res.value > 0.0


def test_alpha_policy():
    """Verify strict alpha channel handling: only strip when fully opaque, require explicit policy otherwise."""
    # 1. Fully opaque RGBA (alpha == 255): safely stripped under default policy
    opaque_ref = np.zeros((64, 64, 4), dtype=np.uint8)
    opaque_ref[..., 3] = 255
    opaque_dist = np.full((64, 64, 4), 10, dtype=np.uint8)
    opaque_dist[..., 3] = 255

    res_opaque = zm.score(opaque_ref, opaque_dist, metric="psnr")
    assert res_opaque.value > 20.0

    # 2. Non-opaque RGBA (translucent): raises ValueError under default policy ('opaque')
    trans_ref = np.zeros((64, 64, 4), dtype=np.uint8)
    trans_ref[..., 3] = 128  # 50% opacity
    trans_dist = np.full((64, 64, 4), 10, dtype=np.uint8)
    trans_dist[..., 3] = 128

    with pytest.raises(ValueError, match="Image contains non-opaque alpha"):
        zm.score(trans_ref, trans_dist, metric="psnr")

    # 3. Explicit matte policies succeed
    res_white = zm.score(trans_ref, trans_dist, metric="psnr", alpha_policy="matte_white")
    assert res_white.value > 20.0

    res_black = zm.score(trans_ref, trans_dist, metric="psnr", alpha_policy="matte_black")
    assert res_black.value > 20.0

    res_comp = zm.score(
        trans_ref,
        trans_dist,
        metric="psnr",
        alpha_policy="composite",
        alpha_bg=(128, 128, 128),
    )
    assert res_comp.value > 20.0

    res_discard = zm.score(trans_ref, trans_dist, metric="psnr", alpha_policy="discard_unchecked")
    assert res_discard.value > 20.0

    # 4. Unknown alpha policy raises ValueError
    with pytest.raises(ValueError, match="Unknown alpha_policy"):
        zm.score(trans_ref, trans_dist, metric="psnr", alpha_policy="unsupported_mode")

    # 5. PIL RGBA handling
    pil_opaque = Image.fromarray(opaque_ref, mode="RGBA")
    pil_opaque_dist = Image.fromarray(opaque_dist, mode="RGBA")
    res_pil_opaque = zm.score(pil_opaque, pil_opaque_dist, metric="psnr")
    assert res_pil_opaque.value > 20.0

    pil_trans = Image.fromarray(trans_ref, mode="RGBA")
    pil_trans_dist = Image.fromarray(trans_dist, mode="RGBA")
    with pytest.raises(ValueError, match="Image contains non-opaque alpha"):
        zm.score(pil_trans, pil_trans_dist, metric="psnr")

    res_pil_matte = zm.score(pil_trans, pil_trans_dist, metric="psnr", alpha_policy="matte_white")
    assert res_pil_matte.value > 20.0


def test_pyiqa_compatibility():
    import torch

    # 1. list_models alias
    models = zm.list_models()
    assert isinstance(models, list)
    assert len(models) > 30

    # 2. create_metric with device, lower_better
    metric = zm.create_metric("ms_ssim", device="cpu")
    assert not metric.lower_better
    assert metric.higher_better
    assert metric.device == "cpu"

    # Chaining
    m_butter = zm.create_metric("butteraugli").cpu()
    assert m_butter.lower_better

    # 3. CVVDP requires explicit display model
    ref_1 = torch.rand(1, 3, 64, 64)
    dist_1 = torch.clamp(ref_1 + 0.05, 0.0, 1.0)
    with pytest.raises(ValueError, match="requires an explicit display model"):
        zm.score(ref_1, dist_1, "cvvdp")

    score_cvvdp = zm.score(ref_1, dist_1, "cvvdp", display_model="standard_4k")
    assert score_cvvdp.item() > 0.0

    # 4. Batched PyTorch tensor scoring (B=3, C=3, H=64, W=64)
    ref_batch = torch.rand(3, 3, 64, 64)
    dist_batch = torch.clamp(ref_batch + 0.05, 0.0, 1.0)
    batch_res = metric(ref_batch, dist_batch)
    assert isinstance(batch_res, zm.BatchScoreResult)
    assert len(batch_res) == 3
    assert float(batch_res) > 0.0
    assert batch_res.item() == batch_res.mean()
    t = batch_res.to_tensor()
    assert t.shape == (3,)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(f"Running {name}...")
            fn()
            print(f"  {name} PASSED")
    print("\nALL UNIT TESTS COMPLETED SUCCESSFULLY!")
