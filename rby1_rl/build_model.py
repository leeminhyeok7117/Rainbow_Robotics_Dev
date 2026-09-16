"""SDK MuJoCo 모델 -> RL용 단일 XML 로 변환."""
import os, xml.etree.ElementTree as ET

SDK = os.path.expanduser("~/rb/rby1-sdk/models/rby1a/mujoco")
SRC = os.path.join(SDK, "model_act.xml")          # position 액추에이터가 이미 정의된 파일
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rby1_reach.xml")

def _walk(el, d, outdir):
    if el.tag == "include":
        return _inline(os.path.join(d, el.get("file")), outdir)
    v = el.get("file")
    if v and not os.path.isabs(v):                 # 메시 경로를 출력 폴더 기준으로 재작성
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

# 1) 베이스를 월드에 고정 (free joint 제거) -> 60kg 부유체 연산 제거 + 드리프트 없음
base = wb.find("./body[@name='base']")
for j in base.findall("joint"):
    if j.get("name") == "world_j":
        base.remove(j); print("removed free joint world_j")

# 2) 오른손 TCP 사이트 (손가락 사이 중심)
def find_body(el, name):
    for b in el.iter("body"):
        if b.get("name") == name: return b
ET.SubElement(find_body(wb, "link_right_arm_6"), "site",
              {"name": "tcp_r", "pos": "0 0 -0.22", "size": "0.012", "rgba": "0 1 0 1"})

# 3) 목표점 = mocap body (물리 없이 위치만 갖는 특수 바디)
#    - geom: 뷰어에서 더블클릭으로 선택 + 드래그 가능하게 함 (contype/conaffinity=0 -> 충돌 없음)
#    - site "target": 기존 env 코드가 mj_name2id(OBJ_SITE, "target") 로 찾던 이름을 그대로 유지
target_body = ET.SubElement(wb, "body", {"name": "target_body", "mocap": "true",
                                         "pos": "0.4 -0.3 1.1"})
ET.SubElement(target_body, "geom", {"type": "sphere", "size": "0.035",
                                    "rgba": "1 0.2 0.2 0.6",
                                    "contype": "0", "conaffinity": "0"})
ET.SubElement(target_body, "site", {"name": "target", "type": "sphere",
                                    "size": "0.035", "rgba": "1 0.2 0.2 0"})

ET.ElementTree(root).write(DST, encoding="unicode")
print("wrote", DST)

# ---------------------------------------------------------------------------
# 토크 제어 변형: 오른팔 7개만 position -> motor 로 교체
# (토르소/왼팔/머리는 position 유지 -> 중력에 무너지지 않고 그대로 서 있음)
# 토크 한계는 실측 기반: 중력 최대 28Nm + 댐핑(50Nm/(rad/s)) 을 함께 이겨야 함
# ---------------------------------------------------------------------------
TORQUE_LIMIT = [150., 150., 100., 100., 50., 50., 50.]
DST_T = DST.replace(".xml", "_torque.xml")

act = root.find("actuator")
for i, lim in enumerate(TORQUE_LIMIT):
    name = f"right_arm_{i+1}_act"
    old = act.find(f"./position[@name='{name}']")
    idx = list(act).index(old)
    joint = old.get("joint")
    act.remove(old)
    act.insert(idx, ET.Element("motor", {"name": name, "joint": joint,
                                         "ctrlrange": f"{-lim} {lim}", "ctrllimited": "true"}))

ET.ElementTree(root).write(DST_T, encoding="unicode")
print("wrote", DST_T, "(오른팔 7개 = motor/토크제어)")
