import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "models", "rby1_reach.xml")  # position 제어만 사용

ARM_JOINTS   = [f"right_arm_{i}" for i in range(7)]
ARM_ACTS     = [f"right_arm_{i+1}_act" for i in range(7)]
TORSO_JOINTS = [f"torso_{i}" for i in range(6)]
TORSO_ACTS   = [f"link{i+1}_act" for i in range(6)]

READY_TORSO = np.deg2rad([0., 0., 0., 20., 0., 0.])
READY_ARM   = np.deg2rad([15., -65., -15., -115., 75., -65., -5.])

# 목표가 튕겨 다닐 상자 (기존 리칭 작업공간보다 살짝 여유를 둠)
TARGET_LOW  = np.array([0.25, -0.55, 0.85])
TARGET_HIGH = np.array([0.60, -0.05, 1.35])

SPEED_RANGE = (0.05, 0.20)  # m/s, 에피소드마다 랜덤


class Rby1TrackEnv(gym.Env):
    """정적 목표가 아니라 상자 안을 등속으로 튕기며 움직이는 목표를 추종한다.

    Rby1ReachEnv 와 관측/보상 구조는 거의 같지만:
      - target 이 매 스텝 이동 (벽에서 반사)
      - 관측에 목표 속도 3차원이 추가됨 (26차원)
      - "도달 후 끝"이 없으므로 보상은 매 스텝 촘촘한 거리항이 핵심
      - success 는 "마지막 순간 근접" 대신 "에피소드 내 근접 상태 유지 비율"로 정의
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, frame_skip=10, max_steps=150,
                 max_delta=0.05, track_radius=0.06, disable_contact=True,
                 manual_target=False):
        # manual_target=True: 목표를 스크립트로 튕기지 않고, 뷰어에서 마우스로
        # 드래그한 mocap body 위치를 매 스텝 그대로 읽어서 target 으로 사용한다.
        self.manual_target = manual_target
        self.model = mujoco.MjModel.from_xml_path(XML)
        if disable_contact:
            self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.data = mujoco.MjData(self.model)

        nid = lambda t, n: mujoco.mj_name2id(self.model, t, n)
        J, A, S = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_SITE
        self.arm_qadr   = np.array([self.model.jnt_qposadr[nid(J, n)] for n in ARM_JOINTS])
        self.arm_vadr   = np.array([self.model.jnt_dofadr[nid(J, n)]  for n in ARM_JOINTS])
        self.arm_act    = np.array([nid(A, n) for n in ARM_ACTS])
        self.torso_qadr = np.array([self.model.jnt_qposadr[nid(J, n)] for n in TORSO_JOINTS])
        self.torso_act  = np.array([nid(A, n) for n in TORSO_ACTS])
        self.tcp_sid    = nid(S, "tcp_r")
        self.target_sid = nid(S, "target")
        self.target_mid = self.model.body_mocapid[nid(mujoco.mjtObj.mjOBJ_BODY, "target_body")]

        self.ctrl_lo = self.model.actuator_ctrlrange[self.arm_act, 0]
        self.ctrl_hi = self.model.actuator_ctrlrange[self.arm_act, 1]

        self.frame_skip, self.max_steps = frame_skip, max_steps
        self.max_delta, self.track_radius = max_delta, track_radius
        self.dt = frame_skip * self.model.opt.timestep  # 목표를 이동시킬 때 쓰는 실제 제어주기
        self.render_mode, self._renderer = render_mode, None

        self.action_space      = spaces.Box(-1., 1., (7,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (26,), np.float32)  # 23 + 목표속도 3

    def _tcp(self):
        return self.data.site_xpos[self.tcp_sid].copy()

    def _obs(self):
        q   = self.data.qpos[self.arm_qadr]
        qd  = self.data.qvel[self.arm_vadr] * 0.1
        tcp = self._tcp()
        return np.concatenate([q, qd, tcp, self.target, tcp - self.target,
                               self.target_vel]).astype(np.float32)

    def _bounce_target(self):
        """등속 이동 + 상자 벽에서 반사. 물리 바디가 아니라 순수 기구학."""
        nxt = self.target + self.target_vel * self.dt
        for k in range(3):
            if nxt[k] < TARGET_LOW[k] or nxt[k] > TARGET_HIGH[k]:
                self.target_vel[k] *= -1.0
                nxt[k] = np.clip(nxt[k], TARGET_LOW[k], TARGET_HIGH[k])
        self.target = nxt
        self.data.mocap_pos[self.target_mid] = self.target

    def _follow_manual_target(self):
        """뷰어에서 마우스로 드래그한 mocap body 위치를 읽어 target/target_vel 갱신."""
        new_target = self.data.mocap_pos[self.target_mid].copy()
        self.target_vel = (new_target - self.target) / self.dt
        self.target = new_target

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # mj_resetData 가 mocap_pos 도 XML 기본값으로 되돌리므로, 수동 드래그 모드에서는
        # 사용자가 옮겨둔 목표 위치를 리셋 전에 챙겨뒀다가 리셋 후 다시 넣어준다.
        dragged = self.data.mocap_pos[self.target_mid].copy() if self.manual_target else None

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.torso_qadr] = READY_TORSO
        self.data.qpos[self.arm_qadr]   = READY_ARM + self.np_random.uniform(-0.1, 0.1, 7)
        self.data.ctrl[:] = 0.
        self.data.ctrl[self.torso_act] = READY_TORSO
        self.data.ctrl[self.arm_act]   = self.data.qpos[self.arm_qadr]

        if self.manual_target:
            self.target = dragged
            self.target_vel = np.zeros(3)
        else:
            self.target = self.np_random.uniform(TARGET_LOW, TARGET_HIGH)
            speed = self.np_random.uniform(*SPEED_RANGE)
            direction = self.np_random.normal(size=3)
            direction /= np.linalg.norm(direction) + 1e-8
            self.target_vel = direction * speed
        self.data.mocap_pos[self.target_mid] = self.target

        mujoco.mj_forward(self.model, self.data)
        self.t, self.prev_a, self.in_radius = 0, np.zeros(7), 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1., 1.)
        tgt = np.clip(self.data.ctrl[self.arm_act] + a * self.max_delta, self.ctrl_lo, self.ctrl_hi)
        self.data.ctrl[self.arm_act] = tgt
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        if self.manual_target:
            self._follow_manual_target()   # 사용자가 드래그한 위치를 그대로 추종
        else:
            self._bounce_target()          # 팔이 한 스텝 움직이는 동안 목표도 같이 이동

        d = float(np.linalg.norm(self._tcp() - self.target))
        close = d < self.track_radius
        self.in_radius += int(close)

        r = (-d
             + (0.3 if close else 0.0)                                     # 근접 유지 보너스 (매 스텝)
             - 0.010  * float(np.sum(a ** 2))
             - 0.005  * float(np.sum((a - self.prev_a) ** 2))
             - 0.0005 * float(np.sum(self.data.qvel[self.arm_vadr] ** 2)))
        self.prev_a, self.t = a, self.t + 1
        done = self.t >= self.max_steps
        info = {"dist": d, "is_success": close,
                "track_ratio": self.in_radius / self.t if done else None}
        return self._obs(), r, False, done, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
