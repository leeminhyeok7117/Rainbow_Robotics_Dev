import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "models", "rby1_pick.xml")

ARM_JOINTS   = [f"right_arm_{i}" for i in range(7)]
ARM_ACTS     = [f"right_arm_{i+1}_act" for i in range(7)]
TORSO_JOINTS = [f"torso_{i}" for i in range(6)]
TORSO_ACTS   = [f"link{i+1}_act" for i in range(6)]

READY_TORSO = np.deg2rad([0., 0., 0., 20., 0., 0.])
READY_ARM   = np.deg2rad([15., -65., -15., -115., 75., -65., -5.])

TABLE_POS  = np.array([0.42, -0.30, 0.95])
TABLE_TOP  = TABLE_POS[2] + 0.02          # 테이블 상판 높이
BLOCK_HALF = 0.02
BLOCK_XY_RANGE = 0.06                      # 테이블 위에서 블록 위치를 무작위로 흩뿌리는 범위
LIFT_HEIGHT = 0.08                         # 이만큼 들어올리면 성공


class Rby1PickEnv(gym.Env):
    """블록을 오른손으로 집어서 LIFT_HEIGHT 만큼 들어올리는 태스크.

    - 접촉 있음 (손가락-블록, 블록-테이블만 계산되도록 build_pick_model.py 에서 마스킹)
    - 그리퍼는 힘(토크) 제어만 존재 (위치 제어 불가) -> 쥐는 힘도 정책이 학습해야 함
    - reward_mode="sparse": 들어올렸을 때만 +1, 그 외엔 거의 0 (탐색 문제를 그대로 노출)
    - reward_mode="dense" : 거리/접촉/높이에 보상을 매 스텝 부여 (보상 성형)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, frame_skip=10, max_steps=200,
                 max_delta=0.05, reward_mode="dense"):
        assert reward_mode in ("sparse", "dense")
        self.reward_mode = reward_mode
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)

        nid = lambda t, n: mujoco.mj_name2id(self.model, t, n)
        J, A, S, B = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR,
                      mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_BODY)
        self.arm_qadr   = np.array([self.model.jnt_qposadr[nid(J, n)] for n in ARM_JOINTS])
        self.arm_vadr   = np.array([self.model.jnt_dofadr[nid(J, n)]  for n in ARM_JOINTS])
        self.arm_act    = np.array([nid(A, n) for n in ARM_ACTS])
        self.torso_qadr = np.array([self.model.jnt_qposadr[nid(J, n)] for n in TORSO_JOINTS])
        self.torso_act  = np.array([nid(A, n) for n in TORSO_ACTS])
        self.tcp_sid    = nid(S, "tcp_r")
        self.finger_act = nid(A, "right_finger_act")
        self.finger_qadr = self.model.jnt_qposadr[nid(J, "gripper_finger_r2")]
        self.block_bid  = nid(B, "block")
        self.block_qadr = self.model.jnt_qposadr[nid(J, "block_free")]
        self.block_vadr = self.model.jnt_dofadr[nid(J, "block_free")]
        # 손가락 충돌 geom id (그립 접촉 보너스 판정용) -- 이름이 없어 body로 찾는다
        self.finger_geom_ids = set()
        for gi in range(self.model.ngeom):
            bname = mujoco.mj_id2name(self.model, B, self.model.geom_bodyid[gi])
            if bname in ("ee_finger_r1", "ee_finger_r2"):
                self.finger_geom_ids.add(gi)
        self.block_geom = nid(mujoco.mjtObj.mjOBJ_GEOM, "block")

        self.ctrl_lo = self.model.actuator_ctrlrange[self.arm_act, 0]
        self.ctrl_hi = self.model.actuator_ctrlrange[self.arm_act, 1]

        self.frame_skip, self.max_steps, self.max_delta = frame_skip, max_steps, max_delta
        self.render_mode, self._renderer = render_mode, None

        self.action_space      = spaces.Box(-1., 1., (8,), np.float32)   # 팔 7 + 그리퍼 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (25,), np.float32)

    def _tcp(self):
        return self.data.site_xpos[self.tcp_sid].copy()

    def _block_pos(self):
        return self.data.xpos[self.block_bid].copy()

    def _grasp_contacts(self):
        """양쪽 손가락이 동시에 블록에 닿아 있는지."""
        touched = set()
        for i in range(self.data.ncon):
            g1, g2 = self.data.contact.geom1[i], self.data.contact.geom2[i]
            if self.block_geom in (g1, g2):
                other = g2 if g1 == self.block_geom else g1
                if other in self.finger_geom_ids:
                    touched.add(other)
        return len(touched)  # 0, 1, 2

    def _obs(self):
        q   = self.data.qpos[self.arm_qadr]
        qd  = self.data.qvel[self.arm_vadr] * 0.1
        g   = self.data.qpos[self.finger_qadr:self.finger_qadr + 1]
        tcp = self._tcp()
        blk = self._block_pos()
        height = np.array([blk[2] - self.block_z0])
        return np.concatenate([q, qd, g, tcp, blk, tcp - blk, height]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.torso_qadr] = READY_TORSO
        self.data.qpos[self.arm_qadr]   = READY_ARM + self.np_random.uniform(-0.1, 0.1, 7)
        self.data.ctrl[:] = 0.
        self.data.ctrl[self.torso_act] = READY_TORSO
        self.data.ctrl[self.arm_act]   = self.data.qpos[self.arm_qadr]

        xy = TABLE_POS[:2] + self.np_random.uniform(-BLOCK_XY_RANGE, BLOCK_XY_RANGE, 2)
        self.data.qpos[self.block_qadr:self.block_qadr + 3] = [xy[0], xy[1],
                                                                TABLE_TOP + BLOCK_HALF + 0.001]
        self.data.qpos[self.block_qadr + 3:self.block_qadr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.block_vadr:self.block_vadr + 6] = 0

        mujoco.mj_forward(self.model, self.data)
        self.block_z0 = self._block_pos()[2]  # 리셋 직후 실제로 안착한 높이를 기준으로 삼음
        self.t, self.prev_a, self.lifted_steps = 0, np.zeros(8), 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1., 1.)
        self.data.ctrl[self.arm_act] = np.clip(
            self.data.ctrl[self.arm_act] + a[:7] * self.max_delta, self.ctrl_lo, self.ctrl_hi)
        self.data.ctrl[self.finger_act] = a[7] * 100.0   # -1=완전히 닫는 방향, +1=여는 방향
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        tcp, blk = self._tcp(), self._block_pos()
        reach_d = float(np.linalg.norm(tcp - blk))
        height = float(blk[2] - self.block_z0)
        grasped = self._grasp_contacts() == 2
        lifted = height > LIFT_HEIGHT
        self.lifted_steps += int(lifted)

        if self.reward_mode == "sparse":
            r = 1.0 if lifted else 0.0
        else:
            r = (-1.0 * reach_d                                 # 손을 블록 가까이
                 + 2.0 * float(grasped)                         # 양쪽 손가락이 동시에 닿으면 보너스
                 + 3.0 * max(0.0, height)                        # 든 높이만큼 보상
                 + (2.0 if lifted else 0.0))                    # 목표 높이 도달 보너스
        r -= (0.01 * float(np.sum(a[:7] ** 2))
              + 0.005 * float(np.sum((a - self.prev_a) ** 2))
              + 0.0005 * float(np.sum(self.data.qvel[self.arm_vadr] ** 2)))

        self.prev_a, self.t = a, self.t + 1
        done = self.t >= self.max_steps
        info = {"reach_dist": reach_d, "height": height, "grasped": grasped,
                "is_success": self.lifted_steps > 10}  # 10스텝(0.2s) 이상 들고 있어야 성공 인정
        return self._obs(), r, False, done, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None
