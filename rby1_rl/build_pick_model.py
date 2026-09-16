"""SDK MuJoCo 모델 -> 블록 잡기(pick) RL용 XML 생성.

rby1_reach.xml(build_model.py)과 같은 평탄화 로직을 재사용하되, 추가로:
  1) 테이블 + 자유물체(블록) 를 작업공간 안에 배치
  2) 접촉을 "전부 끄기"가 아니라 "블록/손가락/테이블만" 켜는 비트마스크 설계
     -> 로봇 자기 몸체(683 메시) 및 바닥과의 충돌검사는 여전히 0건으로 유지하면서
        그립에 필요한 접촉만 계산 (contact ON의 18배 속도손실을 대부분 회피)
"""
import os, xml.etree.ElementTree as ET
import mujoco

SDK = os.path.expanduser("~/rb/rby1-sdk/models/rby1a/mujoco")
SRC = os.path.join(SDK, "model_act.xml")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rby1_pick.xml")

MANIP_BIT = 2  # 블록/손가락/테이블/바닥 전용 충돌 비트 (로봇 몸체의 기존 비트 1과 분리)


def _walk(el, d, outdir):
    if el.tag == "include":
        return _inline(os.path.join(d, el.get("file")), outdir)
    v = el.get("file")
    if v and not os.path.isabs(v):
        el.set("file", os.path.relpath(os.path.join(d, v), outdir))
    kids = []
    for c in list(el):
        el.remove(c); kids.extend(_walk(c, d, outdir))
    for c in kids: el.append(c)
    return [el]


def _inline(path, outdir):
    d = os.path.dirname(os.path.abspath(path))
    out = []
    for c in list(ET.parse(path).getroot()):
        out.extend(_walk(c, d, outdir))
    return out


outdir = os.path.dirname(DST); os.makedirs(outdir, exist_ok=True)
src_root = ET.parse(SRC).getroot()
root = ET.Element("mujoco", src_root.attrib)
for c in list(src_root):
    for e in _walk(c, os.path.dirname(os.path.abspath(SRC)), outdir):
        root.append(e)

wb = root.find("worldbody")


def find_body(el, name):
    for b in el.iter("body"):
        if b.get("name") == name:
            return b


# 1) 베이스 고정 (world_j 제거) -- reach 와 동일
base = wb.find("./body[@name='base']")
for j in base.findall("joint"):
    if j.get("name") == "world_j":
        base.remove(j); print("removed free joint world_j")

# 2) 오른손 TCP 사이트
ET.SubElement(find_body(wb, "link_right_arm_6"), "site",
              {"name": "tcp_r", "pos": "0 0 -0.22", "size": "0.012", "rgba": "0 1 0 1"})

# 3) 바닥: 로봇 몸체와의 충돌검사에서 제외, manip 전용 비트로 이동
floor = wb.find("./geom[@name='floor']")
floor.set("contype", str(MANIP_BIT)); floor.set("conaffinity", str(MANIP_BIT))

# 4) 로봇 자기충돌 전면 비활성화 (오른손 손가락만 예외)
#    실측 결과: contype=1/conaffinity=1(상호 충돌 활성)이 바닥뿐 아니라 base, 양팔,
#    토르소, 머리, 양 손가락 등 30개 바디에 걸쳐 이미 켜져 있었다. 이게 진짜 병목이었고
#    (mesh-mesh 충돌은 원시도형보다 훨씬 비쌈), 바닥만 옮겨서는 속도가 거의 안 나아졌다.
#    다만 이 값들이 XML엔 <default class=...> 로 상속되어 있어 문자열로는 안 잡히므로,
#    한 번 컴파일해서 실제 활성화된 바디 목록을 얻은 뒤 그 바디의 geom을 통째로 덮어쓴다.
KEEP_COLLIDING = {"ee_finger_r1", "ee_finger_r2"}
tmp_path = DST + ".tmp"
ET.ElementTree(root).write(tmp_path, encoding="unicode")
_tmp_model = mujoco.MjModel.from_xml_path(tmp_path)
active_bodies = {
    mujoco.mj_id2name(_tmp_model, mujoco.mjtObj.mjOBJ_BODY, _tmp_model.geom_bodyid[g])
    for g in range(_tmp_model.ngeom)
    if _tmp_model.geom_contype[g] == 1 and _tmp_model.geom_conaffinity[g] == 1
} - {None, "world"}
os.remove(tmp_path)

for body in wb.iter("body"):
    name = body.get("name")
    if name not in active_bodies:
        continue
    for g in body.findall("geom"):
        if g.get("class") == "visual":
            continue
        if name in KEEP_COLLIDING:
            g.set("contype", str(MANIP_BIT)); g.set("conaffinity", str(MANIP_BIT))
        else:
            g.set("contype", "0"); g.set("conaffinity", "0")
# worldbody 직속 geom(=floor) 은 이미 위에서 처리했으니 body 루프 대상에서 제외됨

# 5) 테이블 (고정, 오른팔 작업공간 중심에 위치)
TABLE_POS = [0.42, -0.30, 0.95]
TABLE_HALF = [0.15, 0.15, 0.02]
ET.SubElement(wb, "geom", {
    "name": "table", "type": "box", "pos": " ".join(map(str, TABLE_POS)),
    "size": " ".join(map(str, TABLE_HALF)), "rgba": "0.55 0.4 0.25 1",
    "contype": str(MANIP_BIT), "conaffinity": str(MANIP_BIT),
    "friction": "1.0 0.01 0.001",
})

# 6) 블록 (자유물체, 테이블 위에서 시작 -- 정확한 z 는 env.reset()에서 랜덤화)
BLOCK_HALF = 0.02
block_z = TABLE_POS[2] + TABLE_HALF[2] + BLOCK_HALF
block = ET.SubElement(wb, "body", {"name": "block",
                                   "pos": f"{TABLE_POS[0]} {TABLE_POS[1]} {block_z}"})
ET.SubElement(block, "freejoint", {"name": "block_free"})
ET.SubElement(block, "geom", {
    "name": "block", "type": "box", "size": f"{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}",
    "rgba": "0.9 0.65 0.1 1", "mass": "0.05",
    "contype": str(MANIP_BIT), "conaffinity": str(MANIP_BIT),
    "friction": "1.2 0.01 0.001", "condim": "4",
})

# 7) 목표 표시용 mocap (블록을 들어올려야 할 목표 높이를 시각화, 물리 없음)
goal = ET.SubElement(wb, "body", {"name": "goal_body", "mocap": "true",
                                  "pos": f"{TABLE_POS[0]} {TABLE_POS[1]} {block_z + 0.15}"})
ET.SubElement(goal, "geom", {"type": "sphere", "size": "0.02", "rgba": "0.2 1 0.2 0.5",
                            "contype": "0", "conaffinity": "0"})
ET.SubElement(goal, "site", {"name": "goal", "type": "sphere", "size": "0.02",
                             "rgba": "0.2 1 0.2 0"})

ET.ElementTree(root).write(DST, encoding="unicode")
print("wrote", DST)
