"""保真度阈值标定测试 (遗留项 ①)。

覆盖:
- sweep_fidelity 纯逻辑: 给定标注样本, 选出 F1 最大的拒绝阈值 (并列取最小 t)。
- config 环境变量覆盖: RAG_FIDELITY_THRESHOLD 可无需改码直接生效标定推荐值。
"""

from api.config import Settings
from scripts.calibrate_fidelity import sweep_fidelity


def test_sweep_fidelity_picks_max_f1_threshold():
    """好样本高分(0.8/0.9)、坏样本低分(0.1/0.2): 最佳阈值应卡在 (0.2, 0.22],
    拒掉全部坏样本且不误伤好样本 → F1=1.0。"""
    cases = [
        {"score": 0.9, "is_bad": False},
        {"score": 0.8, "is_bad": False},
        {"score": 0.1, "is_bad": True},
        {"score": 0.2, "is_bad": True},
    ]
    res = sweep_fidelity(cases)
    assert res["best_f1"] == 1.0
    assert res["best_threshold"] == 0.22  # 最小达到 F1=1 的阈值 (step=0.02)


def test_sweep_fidelity_no_rejectable_threshold():
    """全部为好样本 → 任何 t 都会拒绝好样本, F1=0.0, 阈值取 0.0 (默认放行)。"""
    cases = [{"score": 0.9, "is_bad": False}, {"score": 0.8, "is_bad": False}]
    res = sweep_fidelity(cases)
    assert res["best_f1"] == 0.0
    assert res["best_threshold"] == 0.0


def test_config_fidelity_threshold_env_override(monkeypatch):
    """标定脚本产出的推荐值 (如 0.32) 经 RAG_FIDELITY_THRESHOLD 直接生效, 无需改 config.py。"""
    monkeypatch.setenv("RAG_FIDELITY_THRESHOLD", "0.32")
    s = Settings()
    assert s.fidelity_threshold == 0.32


def test_config_fidelity_threshold_default(monkeypatch):
    """未设环境变量且忽略 .env 文件时, 代码默认 0.60。

    pydantic-settings 会从 .env 文件读取, 故实例化时显式 _env_file=None 隔离文件干扰
    (否则 P2 定稿写入 .env 的 0.45 会覆盖代码默认, 与本测试"验证代码默认"的语义无关)。
    """
    monkeypatch.delenv("RAG_FIDELITY_THRESHOLD", raising=False)
    s = Settings(_env_file=None)
    assert s.fidelity_threshold == 0.60
