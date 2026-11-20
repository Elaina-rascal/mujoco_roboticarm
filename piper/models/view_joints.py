"""逐个查看 PiPER XML 的 link 质心和 joint：python3 -m piper.models.view_joints。"""

import argparse
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import mujoco
import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "official": MODEL_DIR / "arm_official.xml",
    "arm": MODEL_DIR / "arm.xml",
}
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 9))
FRAME_COLORS = ([1, 0.1, 0.1, 1], [0.1, 1, 0.1, 1], [0.15, 0.4, 1, 1])
HELP = "Tab link/joint | L link | V joint | M XML | J/K select | [/] move | Z zero | H home | F frames | T transparency | P report"


class JointModel:
    def __init__(self, name):
        self.path = MODEL_PATHS[name]
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        self.joints = np.array([self.model.joint(n).id for n in JOINT_NAMES])
        self.qids = self.model.jnt_qposadr[self.joints]
        names = [*(f"link{i}" for i in range(1, 7)), "gripper_base", "link7", "link8", "base_link"]
        self.links = {
            name: bid for name in names
            if (bid := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
        }
        # 目标球只供 IK 交互使用，这里隐藏它，不修改原 XML。
        target = self.model.body("target")
        start = int(target.geomadr[0])
        self.model.geom_rgba[start:start + int(target.geomnum[0]), 3] = 0
        self.mesh_geoms = np.flatnonzero(self.model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
        self.mesh_alpha = self.model.geom_rgba[self.mesh_geoms, 3].copy()
        self.set_transparent(True)

    def set_transparent(self, enabled):
        self.model.geom_rgba[self.mesh_geoms, 3] = self.mesh_alpha * (0.25 if enabled else 1.0)

    def pose(self, name):
        q = np.zeros(len(JOINT_NAMES))
        if name == "home":
            q = self.model.key("home").qpos[self.qids].copy()
        return q

    def apply(self, q):
        # 只做正运动学，不运行动力学，也不依赖位置执行器跟踪。
        self.data.qpos[self.qids] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def controlling_joint(self, link):
        """固定 body 使用最近的上游关节，底座没有可调关节。"""
        bid = self.links[link]
        while bid:
            if self.model.body_jntnum[bid]:
                matches = np.flatnonzero(self.joints == self.model.body_jntadr[bid])
                return int(matches[0]) if len(matches) else None
            bid = self.model.body_parentid[bid]
        return None

    def report(self):
        m, d = self.model, self.data
        fmt = lambda v: np.array2string(np.asarray(v), precision=6, suppress_small=True)
        print(f"\n===== {self.path.name} =====", flush=True)
        print("pos 单位 m；quat 顺序 wxyz；hinge 的 q/range 单位 deg，slide 单位 m。")
        print("body.pos/quat 相对父 body；joint.pos/axis 位于所属 body 局部坐标系。")
        for index, jid in enumerate(self.joints):
            bid = m.jnt_bodyid[jid]
            parent = m.body(m.body_parentid[bid]).name
            hinge = m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
            scale, unit = (180 / np.pi, "deg") if hinge else (1, "m")
            print(f"{JOINT_NAMES[index]}: {parent} -> {m.body(bid).name}, "
                  f"{'hinge' if hinge else 'slide'}, "
                  f"q={d.qpos[self.qids[index]] * scale:.3f} {unit}, "
                  f"range={fmt(m.jnt_range[jid] * scale)} {unit}")
            print(f"  body.pos={fmt(m.body_pos[bid])}, body.quat={fmt(m.body_quat[bid])}")
            print(f"  joint.pos={fmt(m.jnt_pos[jid])}, joint.axis={fmt(m.jnt_axis[jid])}")
            print(f"  world.anchor={fmt(d.xanchor[jid])}, world.+axis={fmt(d.xaxis[jid])}")
        print("\n各 body 自身的质量和质心（不是下游合成质心）：")
        for name, bid in self.links.items():
            print(f"{name}: mass={m.body_mass[bid]:.6f} kg, "
                  f"local.COM={fmt(m.body_ipos[bid])} m, world.COM={fmt(d.xipos[bid])} m")


def arrow(scene, start, direction, length, width, color, label=""):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                       start, np.eye(3).ravel(), np.asarray(color, dtype=float))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width,
                        start, start + length * direction)
    scene.ngeom += 1
    if label:
        # 标签放在箭头尖端，避免 XYZ 和同原点关节的文字完全重叠。
        marker = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, width), start + length * direction,
                           np.eye(3).ravel(), np.asarray(color, dtype=float))
        marker.label = label
        scene.ngeom += 1


def draw(scene, robot, selected, mode, frames):
    scene.ngeom = 0
    if mode == "joint":
        jid = robot.joints[selected]
        bid = robot.model.jnt_bodyid[jid]
        origin = robot.data.xanchor[jid]
        # xaxis 是实际关节轴，不能用惯性坐标系或可视网格坐标系代替。
        arrow(scene, origin, robot.data.xaxis[jid], 0.18, 0.006,
              [1, 0.8, 0.05, 1], f"{JOINT_NAMES[selected]} +axis")
    else:
        bid = robot.links[selected]
        origin = robot.data.xpos[bid]
        com = robot.data.xipos[bid]
        # xipos 是当前 body 自身惯性坐标原点的世界位置，即自身质心。
        marker = scene.geoms[scene.ngeom]
        color = np.array([1, 0.15, 0.8, 1])
        mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, 0.009), com, np.eye(3).ravel(), color)
        marker.label = f"{selected} COM"
        scene.ngeom += 1
        if np.linalg.norm(com - origin) > 1e-9:
            line = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(line, mujoco.mjtGeom.mjGEOM_LINE,
                               np.zeros(3), origin, np.eye(3).ravel(), color)
            mujoco.mjv_connector(line, mujoco.mjtGeom.mjGEOM_LINE, 2, origin, com)
            scene.ngeom += 1
    if frames:
        rotation = robot.data.xmat[bid].reshape(3, 3)
        for axis, color in enumerate(FRAME_COLORS):
            arrow(scene, origin, rotation[:, axis], 0.09, 0.002, color,
                  f"{robot.model.body(bid).name}.{'XYZ'[axis]}")


def change_joint(q, index, amount, models):
    # 两个模型共用姿态，使用限位交集，避免切换时被悄悄截断。
    low = max(r.model.jnt_range[r.joints[index], 0] for r in models.values())
    high = min(r.model.jnt_range[r.joints[index], 1] for r in models.values())
    q[index] = np.clip(q[index] + amount, low, high)
    if index == 6:
        q[7] = -q[6]
    elif index == 7:
        q[6] = -q[7]


def view(models, active, q, step):
    import mujoco.viewer

    keys = SimpleQueue()
    selected_joint, selected_link = 0, "link1"
    mode, frames, transparent = "link", False, True
    camera = None
    print(HELP, flush=True)
    print("Tab 切换 link / joint，每次仅显示一个对象；J/K 选择下一个/上一个。")
    print("粉色球 = link 自身质心；黄箭头 = joint 正轴；F 开启局部 XYZ。")
    print("两套模型保持相同 q；调整范围使用两者限位交集，双指联动。", flush=True)
    while True:
        robot = models[active]
        if selected_link not in robot.links:
            selected_link = "link6"
        robot.set_transparent(transparent)
        robot.apply(q)
        switch = False
        with mujoco.viewer.launch_passive(
            robot.model, robot.data, key_callback=keys.put,
            show_left_ui=False, show_right_ui=False,
        ) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.18, 0, 0.3]
                viewer.cam.distance = 1.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -20
                if camera is not None:
                    lookat, distance, azimuth, elevation = camera
                    viewer.cam.lookat[:] = lookat
                    viewer.cam.distance = distance
                    viewer.cam.azimuth = azimuth
                    viewer.cam.elevation = elevation
            print(f"\n当前 XML: {robot.path.name}", flush=True)
            while viewer.is_running():
                report = False
                try:
                    while True:
                        key = keys.get_nowait()
                        if key == ord("M"):
                            switch = True
                        elif key == 258:  # GLFW_KEY_TAB
                            mode = "joint" if mode == "link" else "link"
                        elif key in (ord("L"), ord("V")):
                            mode = "link" if key == ord("L") else "joint"
                        elif key in (ord("J"), ord("K")):
                            direction = 1 if key == ord("J") else -1
                            if mode == "joint":
                                selected_joint = (selected_joint + direction) % 8
                            else:
                                names = list(robot.links)
                                selected_link = names[(names.index(selected_link) + direction) % len(names)]
                        elif key in (ord("["), ord("]")):
                            index = (selected_joint if mode == "joint"
                                     else robot.controlling_joint(selected_link))
                            if index is not None:
                                amount = np.deg2rad(step) if index < 6 else 0.001
                                change_joint(q, index, amount * (1 if key == ord("]") else -1), models)
                        elif key in (ord("Z"), ord("H")):
                            q[:] = robot.pose("zero" if key == ord("Z") else "home")
                        elif key == ord("F"):
                            frames = not frames
                        elif key == ord("T"):
                            transparent = not transparent
                        elif key == ord("P"):
                            report = True
                except Empty:
                    pass
                with viewer.lock():
                    # 原生全局关节/质心显示会一次标出所有对象，保持关闭。
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
                    viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
                    robot.set_transparent(transparent)
                    robot.apply(q)
                    if report:
                        for model in models.values():
                            model.apply(q)
                            model.report()
                    selected = selected_joint if mode == "joint" else selected_link
                    draw(viewer.user_scn, robot, selected, mode, frames)
                    if switch:
                        camera = (viewer.cam.lookat.copy(), viewer.cam.distance,
                                  viewer.cam.azimuth, viewer.cam.elevation)
                if switch:
                    break
                if hasattr(viewer, "set_texts"):
                    if mode == "joint":
                        unit = "deg" if selected_joint < 6 else "mm"
                        value = (np.rad2deg(q[selected_joint]) if selected_joint < 6
                                 else q[selected_joint] * 1000)
                        detail = f"joint: {JOINT_NAMES[selected_joint]} = {value:.1f} {unit}\nYellow: joint +axis\n"
                    else:
                        bid = robot.links[selected_link]
                        local = np.array2string(robot.model.body_ipos[bid], precision=4)
                        detail = (f"link: {selected_link} | mass = {robot.model.body_mass[bid]:.6f} kg\n"
                                  f"Local COM (m): {local}\nPink: link COM; line: body origin to COM\n")
                    viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      f"{robot.path.name}\n{detail}"
                                      "Tab: link/joint | L: link | V: joint\n"
                                      "M: XML | J/K: select | [/]: move | Z/H: zero/home\n"
                                      "F: XYZ | T: transparency | P: report", ""))
                viewer.sync()
                time.sleep(1 / 60)
        if not switch:
            return
        active = "arm" if active == "official" else "official"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_PATHS, default="official", help="首先显示的 XML")
    parser.add_argument("--pose", choices=("zero", "home"), default="zero", help="初始姿态")
    parser.add_argument("--step", type=float, default=5.0, help="每次转动的角度，单位 deg")
    parser.add_argument("--report", action="store_true", help="打印两套 XML 的关节及 link 质心，不打开窗口")
    args = parser.parse_args()
    if not np.isfinite(args.step) or args.step <= 0:
        parser.error("--step 必须是有限的正数")
    models = {name: JointModel(name) for name in MODEL_PATHS}
    q = models[args.model].pose(args.pose)
    if args.report:
        for robot in models.values():
            robot.apply(q)
            robot.report()
        return
    try:
        view(models, args.model, q, args.step)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
