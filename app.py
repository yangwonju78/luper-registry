import base64
import math
import os
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

db_url = os.getenv("DATABASE_URL", "sqlite:///registry.db")
if db_url.startswith("postgres://"):
    db_url = "postgresql+psycopg://" + db_url[len("postgres://"):]
elif db_url.startswith("postgresql://") and "+psycopg" not in db_url:
    db_url = "postgresql+psycopg://" + db_url[len("postgresql://"):]

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

db = SQLAlchemy(app)
ADMIN_KEY = os.getenv("ADMIN_KEY", "change-me")
VERSION = "0.3.0"

class Link(db.Model):
    __tablename__ = "links"
    id = db.Column(db.Integer, primary_key=True)
    key_text = db.Column(db.String(300), nullable=False, index=True)
    display_name = db.Column(db.String(300), nullable=False)
    url = db.Column(db.Text, nullable=False)
    use_location = db.Column(db.Boolean, nullable=False, default=False)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    radius_m = db.Column(db.Float, nullable=False, default=150.0)
    priority = db.Column(db.Integer, nullable=False, default=0)
    enabled = db.Column(db.Boolean, nullable=False, default=True)
    visual_threshold = db.Column(db.Float, nullable=False, default=70.0)
    reference_image = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    def public_dict(self):
        return {
            "id": self.id, "key_text": self.key_text, "display_name": self.display_name,
            "url": self.url, "use_location": bool(self.use_location),
            "latitude": self.latitude, "longitude": self.longitude,
            "radius_m": self.radius_m, "priority": self.priority,
            "enabled": bool(self.enabled), "visual_threshold": self.visual_threshold,
            "reference_image": True,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

def norm(s):
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum())

def hav(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def b64_bytes(data):
    if not data:
        return None
    raw = data.split(",", 1)[1] if "," in data else data
    try:
        return base64.b64decode(raw, validate=False)
    except Exception:
        return None

def decode_gray_bytes(raw):
    if not raw:
        return None
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

def visual_score(reference_bytes, frame_bytes):
    ref = decode_gray_bytes(reference_bytes)
    frame = decode_gray_bytes(frame_bytes)
    if ref is None or frame is None:
        return 0.0
    orb = cv2.ORB_create(nfeatures=1400, fastThreshold=8)
    k1, d1 = orb.detectAndCompute(ref, None)
    k2, d2 = orb.detectAndCompute(frame, None)
    if d1 is None or d2 is None or len(k1) < 6 or len(k2) < 6:
        return 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.72 * n.distance]
    if len(good) < 4:
        return min(35.0, len(good) * 8.0)
    coverage = min(1.0, len(good) / max(12.0, min(len(k1), 40.0)))
    inlier_ratio = 0.0
    if len(good) >= 6:
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is not None and len(mask):
                inlier_ratio = float(mask.sum()) / len(mask)
        except cv2.error:
            pass
    return round(100.0 * (0.58 * coverage + 0.42 * inlier_ratio), 1)

def require_admin():
    return bool(ADMIN_KEY) and request.headers.get("X-Admin-Key", "") == ADMIN_KEY

INDEX_HTML = '<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n<title>LUPER Registry V0.3</title>\n<style>\nbody{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f5f6f8;color:#17191c}\n.w{max-width:1000px;margin:28px auto;padding:0 16px}.c{background:#fff;border:1px solid #ddd;border-radius:14px;padding:18px;margin:14px 0}\n.g{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{display:block;font-size:13px;color:#555;margin:0 0 5px}\ninput{width:100%;box-sizing:border-box;padding:10px;border:1px solid #cfd4da;border-radius:8px;font-size:15px}\n.full{grid-column:1/-1}.row{display:flex;gap:8px;align-items:center}.row input{width:auto}\nbutton{padding:11px 14px;border:0;border-radius:8px;background:#111;color:#fff;font-weight:700;cursor:pointer}\ntable{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}\n.warn{color:#a33;font-weight:700}.ok{color:#176b36;font-weight:700}.muted{color:#666;font-size:13px}\n@media(max-width:700px){.g{grid-template-columns:1fr}.full{grid-column:auto}table{display:block;overflow:auto}}\n</style>\n</head>\n<body><div class="w">\n<h2>LUPER Visual Link Registry V0.3</h2>\n<p>외부 Registry · <b>TEXT + VISUAL + 선택적 GPS</b>. <span class="warn">문자만 같아서는 링크되지 않습니다.</span></p>\n\n<div class="c">\n<label>관리자 키</label>\n<div class="row"><input id="admin" type="password" placeholder="배포 서버의 ADMIN_KEY" style="flex:1"><button type="button" id="saveKey">키 저장</button></div>\n<p class="muted">키는 이 브라우저 탭의 sessionStorage에만 저장됩니다.</p>\n</div>\n\n<div class="c"><form id="f"><div class="g">\n<div><label>등록 문자열</label><input id="key" required placeholder="LUPER사용설명서"></div>\n<div><label>표시명</label><input id="name" required placeholder="LUPER 사용설명서"></div>\n<div class="full"><label>URL</label><input id="url" required placeholder="https://..."></div>\n<div class="full"><label>실제 글자/간판 기준 이미지 (필수)</label><input id="img" type="file" accept="image/*" required></div>\n<div><label>시각 매칭 최소점수</label><input id="thr" type="number" min="0" max="100" value="70"></div>\n<div><label>우선순위</label><input id="pri" type="number" value="0"></div>\n<div class="full row"><input id="use" type="checkbox"><label for="use" style="margin:0">GPS 조건 사용</label></div>\n<div><label>위도</label><input id="lat" type="number" step="any"></div>\n<div><label>경도</label><input id="lon" type="number" step="any"></div>\n<div><label>반경(m)</label><input id="rad" type="number" value="150"></div>\n<div class="full"><button>등록</button></div>\n</div></form></div>\n\n<div class="c"><div id="state" class="muted">서버 확인 중…</div><div id="list"></div></div>\n<div class="c">판정: OCR 후보 → 등록문자 일치 → 등록 기준이미지 시각매칭 → 임계값 통과 → GPS 조건 → 후보 정렬</div>\n</div>\n<script>\nconst $=id=>document.getElementById(id);\nconst esc=s=>String(s??\'\').replace(/[&<>\\"]/g,c=>({\'&\':\'&amp;\',\'<\':\'&lt;\',\'>\':\'&gt;\',\'"\':\'&quot;\'}[c]));\n$(\'admin\').value=sessionStorage.getItem(\'luper_admin_key\')||\'\';\n$(\'saveKey\').addEventListener(\'click\',()=>{sessionStorage.setItem(\'luper_admin_key\',$(\'admin\').value);alert(\'이 탭에 관리자 키를 저장했습니다.\')});\nconst adminHeaders=()=>({\'X-Admin-Key\':$(\'admin\').value});\n\nasync function refresh(){\n  try{\n    const hr=await fetch(\'/health\'); const hs=await hr.json();\n    $(\'state\').innerHTML=`<span class="ok">ONLINE</span> · V${esc(hs.version)} · DB ${esc(hs.database)}`;\n    const r=await fetch(\'/api/entries\',{headers:adminHeaders()});\n    if(r.status===401){$(\'list\').innerHTML=\'<p class="warn">관리자 키를 입력하면 등록목록이 보입니다.</p>\';return}\n    const a=await r.json();\n    let h=\'<table><tr><th>문자</th><th>표시명</th><th>Visual</th><th>GPS</th><th></th></tr>\';\n    for(const x of a)h+=`<tr><td><b>${esc(x.key_text)}</b></td><td>${esc(x.display_name)}</td><td>${x.visual_threshold}%</td><td>${x.use_location?x.radius_m+\'m\':\'무관\'}</td><td><button onclick="delx(${x.id})">삭제</button></td></tr>`;\n    $(\'list\').innerHTML=h+\'</table>\';\n  }catch(e){$(\'state\').innerHTML=\'<span class="warn">서버 연결 실패</span>\'}\n}\nasync function delx(id){\n  const r=await fetch(\'/api/entries/\'+id,{method:\'DELETE\',headers:adminHeaders()});\n  if(!r.ok){alert(await r.text());return} refresh()\n}\nfunction file64(f){return new Promise((ok,no)=>{let r=new FileReader();r.onload=()=>ok(r.result);r.onerror=no;r.readAsDataURL(f)})}\n$(\'f\').addEventListener(\'submit\',async e=>{\n  e.preventDefault();\n  if(!$(\'admin\').value){alert(\'관리자 키를 먼저 입력하세요.\');return}\n  const image=await file64($(\'img\').files[0]);\n  const o={key_text:$(\'key\').value,display_name:$(\'name\').value,url:$(\'url\').value,reference_image_base64:image,\n    visual_threshold:+$(\'thr\').value||70,use_location:$(\'use\').checked,\n    latitude:$(\'lat\').value?+$(\'lat\').value:null,longitude:$(\'lon\').value?+$(\'lon\').value:null,\n    radius_m:+$(\'rad\').value||150,priority:+$(\'pri\').value||0};\n  const r=await fetch(\'/api/entries\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\',\'X-Admin-Key\':$(\'admin\').value},body:JSON.stringify(o)});\n  if(!r.ok){alert(await r.text());return}\n  $(\'f\').reset();$(\'thr\').value=70;$(\'rad\').value=150;$(\'pri\').value=0;refresh()\n});\nrefresh();\n</script></body></html>'

@app.get("/")
def index():
    return Response(INDEX_HTML, content_type="text/html; charset=utf-8")

@app.get("/health")
def health():
    return jsonify(ok=True, version=VERSION, database=db.engine.url.get_backend_name())

@app.get("/api/entries")
def entries():
    if not require_admin():
        return jsonify(error="unauthorized"), 401
    rows = Link.query.filter_by(enabled=True).order_by(Link.priority.desc(), Link.id.desc()).all()
    return jsonify([x.public_dict() for x in rows])

@app.post("/api/entries")
def create_entry():
    if not require_admin():
        return jsonify(error="unauthorized"), 401
    x = request.get_json(silent=True) or {}
    if not all(x.get(k) for k in ("key_text", "display_name", "url", "reference_image_base64")):
        return jsonify(error="문자/표시명/URL/기준이미지는 필수입니다."), 400
    raw = b64_bytes(x.get("reference_image_base64"))
    if not raw or decode_gray_bytes(raw) is None:
        return jsonify(error="기준 이미지를 읽을 수 없습니다."), 400
    use_location = bool(x.get("use_location"))
    if use_location and (x.get("latitude") is None or x.get("longitude") is None):
        return jsonify(error="GPS 사용 시 위도/경도가 필요합니다."), 400
    item = Link(
        key_text=str(x["key_text"]).strip(), display_name=str(x["display_name"]).strip(),
        url=str(x["url"]).strip(), reference_image=raw,
        visual_threshold=float(x.get("visual_threshold") or 70),
        use_location=use_location,
        latitude=float(x["latitude"]) if x.get("latitude") is not None else None,
        longitude=float(x["longitude"]) if x.get("longitude") is not None else None,
        radius_m=float(x.get("radius_m") or 150), priority=int(x.get("priority") or 0),
    )
    db.session.add(item); db.session.commit()
    return jsonify(item.public_dict()), 201

@app.delete("/api/entries/<int:item_id>")
def delete_entry(item_id):
    if not require_admin():
        return jsonify(error="unauthorized"), 401
    item = db.session.get(Link, item_id)
    if not item:
        return jsonify(error="not found"), 404
    db.session.delete(item); db.session.commit()
    return jsonify(ok=True)

@app.post("/api/match_visual")
def match_visual():
    x = request.get_json(silent=True) or {}
    text = str(x.get("text", ""))
    nt = norm(text)
    if len(nt) < 2:
        return jsonify(matches=[])
    frame_bytes = b64_bytes(x.get("frame_jpeg_base64"))
    if not frame_bytes:
        return jsonify(error="frame required"), 400
    lat, lon = x.get("lat"), x.get("lon")
    out = []
    for r in Link.query.filter_by(enabled=True).all():
        nk = norm(r.key_text)
        if not nk or nk not in nt:
            continue
        vs = visual_score(r.reference_image, frame_bytes)
        if vs < float(r.visual_threshold or 70):
            continue
        dist = None
        if r.use_location:
            if lat is None or lon is None or r.latitude is None or r.longitude is None:
                continue
            dist = hav(float(lat), float(lon), r.latitude, r.longitude)
            if dist > r.radius_m:
                continue
        loc_bonus = max(0.0, 100.0 - (dist / r.radius_m * 100.0)) if dist is not None and r.radius_m else 0.0
        score = round(vs * 10 + len(nk) * 2 + int(r.priority) * 100 + loc_bonus, 1)
        d = r.public_dict()
        d["visual_score"] = vs
        d["distance_m"] = round(dist, 1) if dist is not None else None
        d["score"] = score
        out.append(d)
    out.sort(key=lambda z: (z["score"], len(norm(z["key_text"]))), reverse=True)
    return jsonify(matches=out, version=VERSION)

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8788"))
    app.run(host="0.0.0.0", port=port, debug=False)
