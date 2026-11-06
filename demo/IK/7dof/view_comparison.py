"""将离线关键帧日志恢复到已启动的 Visdom 服务器。"""

import argparse
from pathlib import Path

from visdom import Visdom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, nargs="?", help="默认打开最近的 Visdom 日志")
    parser.add_argument("--server", default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    path = args.log
    if path is None:
        files = sorted(Path(__file__).with_name("log").glob("*.visdom.jsonl"))
        if not files:
            parser.error("尚无记录，请先运行仿真")
        path = files[-1]
    if not path.is_file():
        parser.error(f"日志不存在: {path}")
    vis = Visdom(server=args.server, port=args.port, use_incoming_socket=False,
                 use_preflight_checks=False)
    if not vis.check_connection(timeout_seconds=2):
        parser.error("Visdom 未启动，请先运行 python -m visdom.server -port 8097")
    # 日志包含初次创建面板事件，重新回放会先重置曲线，避免重复追加。
    vis.replay_log(str(path))
    print(f"已恢复: {path}\n查看: {args.server}:{args.port}")


if __name__ == "__main__":
    main()
