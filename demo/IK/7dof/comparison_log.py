"""Visdom 离线关键帧记录；运行控制算法时不连接可视化服务器。"""

from datetime import datetime
from pathlib import Path

import numpy as np
from visdom import Visdom


class ComparisonRecorder:
    def __init__(self, every=10):
        if every < 1:
            raise ValueError("关键帧间隔必须至少为 1")
        self.every = every
        folder = Path(__file__).with_name("log")
        folder.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = folder / f"res_vs_dq_{stamp}.visdom.jsonl"
        self.env = f"res_vs_dq_{stamp}"
        self.vis = Visdom(
            env=self.env, offline=True, log_to_filename=str(self.path),
            use_incoming_socket=False, use_preflight_checks=False,
        )
        self.step = 0
        self.keyframes = 0
        self.pending = []
        self.last = None
        self.started = False
        print(f"Visdom 离线关键帧: {self.path}")

    def log(self, iteration, res, dq):
        # 只复制两个七维数组，保留末帧；不在这里做网络请求或绘图。
        self.last = (self.step, res.copy(), dq.copy())
        self.step += 1
        if iteration % self.every == 0:
            self.pending.append(self.last)

    def flush(self):
        """求解结束后一次性记录这批关键帧，并补上末帧。"""
        if self.last is None:
            return
        if not self.pending or self.pending[-1][0] != self.last[0]:
            self.pending.append(self.last)
        steps = np.array([frame[0] for frame in self.pending])
        values = np.array([[frame[1], frame[2]] for frame in self.pending])
        update = "append" if self.started else None
        for joint in range(7):
            self.vis.line(
                X=np.repeat(steps[:, None], 2, axis=1), Y=values[:, :, joint],
                win=f"joint{joint + 1}", update=update,
                opts=dict(title=f"joint{joint + 1}: res vs dq", legend=["res", "dq"],
                          xlabel="IK iteration (global)", ylabel="increment before x0.1"),
            )
        self.vis.line(
            X=steps, Y=np.linalg.norm(values[:, 0] - values[:, 1], axis=1),
            win="difference_norm", update=update,
            opts=dict(title="||res - dq||", xlabel="IK iteration (global)"),
        )
        _, res, dq = self.last
        self.vis.heatmap(
            np.column_stack([res, dq, res - dq]), win="latest_arrays",
            opts=dict(title=f"Last keyframe: {self.last[0]}",
                      rownames=[f"joint{i + 1}" for i in range(7)],
                      columnnames=["res", "dq", "res - dq"]),
        )
        self.keyframes += len(self.pending)
        self.pending.clear()
        self.last = None
        self.started = True
