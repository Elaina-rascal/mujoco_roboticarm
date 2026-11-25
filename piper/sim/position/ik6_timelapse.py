"""完整六自由度 IK；保存每次迭代的 PNG、JSON 轨迹和 GIF，不连接真机。"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
from PIL import Image, ImageDraw, ImageFont

from piper.controllers.pinocchio_controller import PinocchioControl
from piper.interface.base import vector
from piper.models import OFFICIAL_MODEL


INITIAL_Q = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])
TARGET_POSITION = np.array([0.249335, 0.0308932, 0.266156])


def solve_ik6(control, initial_q, target_position, target_rotation, max_iterations=100,
              check=False, stop_on_limit=False, rank_tolerance=1e-3, step_tolerance=1e-4,
              post_rank_iterations=20):
    """沿用现有 IK 的 DLS、0.1 步长和关节限位，完整约束六维误差。

    check=False 跳过奇异值和最大关节变化检查，但仍记录这些数值。
    不调用现有五维 ik()，因为其 H 固定放开 yaw。
    每次更新前记录一次状态，包含初始状态与最后一次更新后的状态。
    """
    q_initial = vector(initial_q, 6).copy()
    target = vector(target_position, 3)
    target_rotation = np.asarray(target_rotation, dtype=float)
    if (target_rotation.shape != (3, 3) or not np.all(np.isfinite(target_rotation))
            or not np.allclose(target_rotation.T @ target_rotation, np.eye(3))
            or not np.isclose(np.linalg.det(target_rotation), 1.0)):
        raise ValueError('target_rotation 必须为旋转矩阵')
    if max_iterations < 0:
        raise ValueError('max_iterations 必须非负')
    if not isinstance(post_rank_iterations, int) or post_rank_iterations < 0:
        raise ValueError('post_rank_iterations 必须为非负整数')
    if (not np.isfinite(rank_tolerance) or rank_tolerance <= 0
            or not np.isfinite(step_tolerance) or step_tolerance <= 0):
        raise ValueError('rank_tolerance 和 step_tolerance 必须为正有限数')
    q = q_initial.copy()
    frames = []
    stop_reason = 'iteration_budget'
    stagnant_steps = 0
    first_rank_loss = None
    target_pose = pin.SE3(target_rotation, target)
    for iteration in range(max_iterations + 1):
        pose = control.fk(q)
        relative = pose.rotation @ target_rotation.T
        error = np.r_[target - pose.translation, -pin.log3(relative)]
        jacobian = pin.computeFrameJacobian(
            control.model, control.data, q, control.ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).copy()
        rate_map = np.eye(6)
        rate_map[3:, 3:] = pin.Jlog3(relative.T)
        jacobian = rate_map @ jacobian  # 六行全部保留，无五维 H 投影。
        u, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
        change = pin.difference(control.model, q_initial, q)
        step_change = (float(np.max(np.abs(q - np.array(frames[-1]['q']))))
                       if frames else None)
        effective_rank = int(np.count_nonzero(singular_values > rank_tolerance))
        if effective_rank < 6 and first_rank_loss is None:
            first_rank_loss = iteration
        log6_error = pin.log6(pose.actInv(target_pose)).vector.copy()
        frames.append(dict(
            iteration=iteration, q=q.tolist(), position=pose.translation.tolist(),
            rotation=pose.rotation.tolist(), error=error.tolist(),
            log6_error=log6_error.tolist(), log6_error_norm=float(np.linalg.norm(log6_error)),
            position_error=float(np.linalg.norm(error[:3])),
            rotation_error=float(np.linalg.norm(error[3:])),
            residual=float(np.linalg.norm(error)),
            min_singular_value=float(singular_values[-1]),
            singular_values=singular_values.tolist(), effective_rank=effective_rank,
            step_change=step_change,
            max_joint_change_deg=float(np.rad2deg(np.max(np.abs(change)))),
            joints_at_limit=(np.flatnonzero(
                np.isclose(q, control.model.lowerPositionLimit, atol=1e-7)
                | np.isclose(q, control.model.upperPositionLimit, atol=1e-7)
            ) + 1).tolist(),
        ))
        if np.linalg.norm(error) <= control.max_task_error:
            stop_reason = 'converged'
            break
        if (stop_on_limit and first_rank_loss is not None
                and iteration - first_rank_loss >= post_rank_iterations):
            stop_reason = 'post_rank_loss'
            break
        stagnant_steps = stagnant_steps + 1 if step_change is not None and step_change < step_tolerance else 0
        if stop_on_limit and stagnant_steps >= 2:
            stop_reason = 'stagnation'
            break
        if iteration == max_iterations:
            break
        gains = singular_values / (singular_values**2 + control.damping**2)
        delta_q = vt.T @ (gains * (u.T @ error))
        q = np.clip(pin.integrate(control.model, q, 0.1 * delta_q),
                    control.model.lowerPositionLimit, control.model.upperPositionLimit)
    converged = frames[-1]['residual'] <= control.max_task_error
    accepted = converged and (not check or (
        frames[-1]['min_singular_value'] >= control.min_singular_value
        and np.deg2rad(frames[-1]['max_joint_change_deg']) <= control.max_joint_change
    ))
    return dict(
        check=check, task_dimension=6, damping=control.damping, step_size=0.1,
        tolerance=control.max_task_error, max_iterations=max_iterations,
        initial_q=q_initial.tolist(), target_position=target.tolist(),
        target_rotation=target_rotation.tolist(), converged=converged,
        accepted=accepted, frames=frames, stop_reason=stop_reason,
        rank_tolerance=rank_tolerance, step_tolerance=step_tolerance,
        first_rank_loss=first_rank_loss, post_rank_iterations=post_rank_iterations,
    )


def add_line(scene, start, end, color, width=0.003):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                       np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end)
    scene.ngeom += 1


def add_axes(scene, position, rotation, length):
    for axis, color in enumerate(([1, .15, .15, 1], [.15, 1, .15, 1], [.2, .45, 1, 1])):
        add_line(scene, position, position + length * rotation[:, axis], color)


def render_trace(trace, output):
    output = Path(output)
    frame_dir = output / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(OFFICIAL_MODEL))
    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 540
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    q_ids = [model.joint(f'joint{i}').qposadr[0] for i in range(1, 7)]
    target_id = int(model.body('target').mocapid[0])
    target = np.array(trace['target_position'])
    rotation = np.array(trace['target_rotation'])
    data.mocap_pos[target_id] = target
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    # 轻微向下俯视，保留侧向观察并看清末端姿态。
    camera.lookat[:] = [.05, .10, .30]
    camera.distance = 1.05
    camera.azimuth = 240
    camera.elevation = -25
    font = ImageFont.load_default(size=16)
    joint_limits = model.jnt_range[[model.joint(f'joint{i}').id for i in range(1, 7)]]
    images = []
    with mujoco.Renderer(model, height=540, width=720) as renderer:
        for index, state in enumerate(trace['frames']):
            data.qpos[q_ids] = state['q']
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            scene = renderer.scene
            # 目标点为绿色球，目标与实际姿态均画 XYZ 轴，完整体现六维目标。
            add_axes(scene, target, rotation, .10)
            add_axes(scene, np.array(state['position']), np.array(state['rotation']), .065)
            for previous, current in zip(trace['frames'][:index], trace['frames'][1:index + 1]):
                add_line(scene, np.array(previous['position']), np.array(current['position']),
                         [1, .65, .1, 1], width=.0015)
            image = Image.new('RGB', (720, 580), '#101925')
            image.paste(Image.fromarray(renderer.render()), (0, 40))
            draw = ImageDraw.Draw(image)
            q = np.asarray(state['q'])
            margins = np.rad2deg(np.minimum(q - joint_limits[:, 0], joint_limits[:, 1] - q))
            near = [f'J{i+1}:{margin:.1f}°' for i, margin in enumerate(margins) if margin <= 5.0]
            status = 'Limits(<5°): ' + (' '.join(near) if near else 'none')
            sig = f'sig_min={state["min_singular_value"]:.6f}'
            # 始终只用一行，极端情况下所有关节都接近限位时自动缩小字体。
            line_font = font
            while draw.textlength(status + '  |  ' + sig, font=line_font) > 692:
                line_font = ImageFont.load_default(size=line_font.size - 1)
            draw.text((14, 11), status, font=line_font, fill='#ffbd69' if near else '#c9d5e3')
            offset = 14 + draw.textlength(status + '  |  ', font=line_font)
            fill = '#ff7c85' if state['min_singular_value'] <= trace['rank_tolerance'] else '#66e0a3'
            draw.text((offset, 11), sig, font=line_font, fill=fill)
            image.save(frame_dir / f'iteration_{index:04d}.png')
            images.append(image)
    durations = [100] * len(images)
    durations[0] = 800
    durations[-1] = 2500
    gif = output / 'piper_ik6.gif'
    images[0].save(gif, save_all=True, append_images=images[1:], duration=durations, loop=0)
    return gif


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/ik6'))
    parser.add_argument('--target', nargs=3, type=float, default=TARGET_POSITION.tolist())
    parser.add_argument('--rpy-deg', nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument('--initial-q', nargs=6, type=float, default=INITIAL_Q.tolist())
    parser.add_argument('--max-iterations', type=int, default=1000, help='仅作为计算预算兜底')
    parser.add_argument('--rank-tolerance', type=float, default=1e-3)
    parser.add_argument('--step-tolerance', type=float, default=1e-4)
    parser.add_argument('--post-rank-iterations', type=int, default=20, help='首次降秩后额外迭代次数')
    args = parser.parse_args()
    control = PinocchioControl(model_path=OFFICIAL_MODEL, gripper_width=.04)
    trace = solve_ik6(control, args.initial_q, args.target,
                      pin.rpy.rpyToMatrix(np.deg2rad(args.rpy_deg)), args.max_iterations, check=False,
                      stop_on_limit=True, rank_tolerance=args.rank_tolerance, step_tolerance=args.step_tolerance,
                      post_rank_iterations=args.post_rank_iterations)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
    gif = render_trace(trace, args.output)
    print(json.dumps(dict(gif=str(gif), iterations=len(trace['frames']) - 1,
                          converged=trace['converged'], final=trace['frames'][-1]), indent=2))


if __name__ == '__main__':
    main()
