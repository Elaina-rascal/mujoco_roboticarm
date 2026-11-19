"""PiPER 模型路径；XML 和网格资源统一保存在本包。"""

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
ARM_MODEL = MODEL_DIR / "arm.xml"
OFFICIAL_MODEL = MODEL_DIR / "arm_official.xml"
