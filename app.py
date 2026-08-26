import base64
import difflib
import json
import math
import os
import re
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text as sql_text

app = Flask(__name__)

db_url = os.getenv("DATABASE_URL", "sqlite:///registry.db")
if db_url.startswith("postgres://"):
    db_url = "postgresql+psycopg://" + db_url[len("postgres://"):]
elif db_url.startswith("postgresql://") and "+psycopg" not in db_url:
    db_url = "postgresql+psycopg://" + db_url[len("postgresql://"):]

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

db = SQLAlchemy(app)
ADMIN_KEY = os.getenv("ADMIN_KEY", "change-me")
VERSION = "0.7.2"


class Link(db.Model):
    __tablename__ = "links"

    id = db.Column(db.Integer, primary_key=True)
    key_text = db.Column(db.String(300), nullable=False, default="")
    registration_name = db.Column(db.String(300))
    major_category = db.Column(db.String(300))
    minor_category = db.Column(db.String(500))
    recognition_text = db.Column(db.String(1500))
    display_name = db.Column(db.String(300), nullable=False)
    url = db.Column(db.Text, nullable=False)

    match_mode = db.Column(db.String(30), nullable=False, default="auto")
    use_location = db.Column(db.Boolean, nullable=False, default=False)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    radius_m = db.Column(db.Float, nullable=False, default=150.0)
    priority = db.Column(db.Integer, nullable=False, default=0)
    enabled = db.Column(db.Boolean, nullable=False, default=True)

    reference_image = db.Column(db.LargeBinary, nullable=False)
    major_reference = db.Column(db.LargeBinary)
    minor_reference = db.Column(db.LargeBinary)

    visual_threshold = db.Column(db.Float, nullable=False, default=48.0)
    identity_profile = db.Column(db.Text)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def fields(self):
        major = (self.major_category or self.key_text or "").strip()
        minor = (self.minor_category or "").strip()
        return {
            "registration_name": (self.registration_name or self.display_name or major).strip(),
            "major_category": major,
            "minor_category": minor,
            "recognition_text": (self.recognition_text or build_recognition_text(major, minor)).strip(),
            "display_name": (self.display_name or major).strip(),
        }

    def public_dict(self):
        f = self.fields()
        try:
            profile = json.loads(self.identity_profile or "{}")
        except Exception:
            profile = {}
        return {
            "id": self.id,
            **f,
            "url": self.url,
            "match_mode": self.match_mode or "auto",
            "use_location": bool(self.use_location),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "radius_m": self.radius_m,
            "priority": self.priority,
            "enabled": bool(self.enabled),
            "visual_threshold": self.visual_threshold,
            "profile_summary": {
                "mode": profile.get("mode"),
                "major_typography": profile.get("major_typography", {}).get("font_family_probabilities", {}),
                "minor_typography": profile.get("minor_typography", {}).get("font_family_probabilities", {}),
                "color": profile.get("full_visual", {}).get("color", {}),
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }



class LiveTrace(db.Model):
    __tablename__ = "live_traces"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(80), nullable=False, index=True)
    event = db.Column(db.String(80), nullable=False)
    payload = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def public_dict(self):
        try:
            p = json.loads(self.payload or "{}")
        except Exception:
            p = {}
        return {
            "id": self.id,
            "session_id": self.session_id,
            "event": self.event,
            "payload": p,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# -------------------------- text / hangul --------------------------

def norm(s):
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum())


def split_terms(s):
    return [x for x in re.split(r"[\s,|/·]+", s or "") if norm(x)]


def build_recognition_terms(major, minor):
    major = (major or "").strip()
    minors = split_terms(minor)
    terms = []
    if major:
        terms.append(major)
    terms.extend(minors)
    for m in minors:
        if major:
            terms.append(f"{major} {m}")
            terms.append(f"{major}{m}")
    if major and minor:
        terms.append(f"{major} {minor}".strip())
        terms.append(f"{major}{minor}".replace(" ", ""))
    # unique preserving order
    return list(dict.fromkeys(t for t in terms if t))


def build_recognition_text(major, minor):
    return " | ".join(build_recognition_terms(major, minor))


CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def decompose_hangul(s):
    out = []
    for ch in norm(s):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            v = code - 0xAC00
            out.extend([CHO[v // 588], JUNG[(v % 588) // 28]])
            jong = JONG[v % 28]
            if jong:
                out.append(jong)
        else:
            out.append(ch)
    return "".join(out)


def sequence_ratio(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()


def hangul_similarity(a, b):
    raw = sequence_ratio(a, b)
    ja = decompose_hangul(a)
    jb = decompose_hangul(b)
    jamo = 100.0 * difflib.SequenceMatcher(None, ja, jb).ratio() if ja and jb else raw
    # OCR "족발→측발" 같은 한두 자 오독을 살리기 위해 자모 비교를 더 신뢰.
    return round(max(raw, raw * 0.35 + jamo * 0.65), 1)


def best_term_score(ocr, target):
    if not target:
        return 0.0
    scores = [hangul_similarity(ocr, target)]
    scores.extend(hangul_similarity(ocr, t) for t in split_terms(target))
    return max(scores, default=0.0)


# -------------------------- image helpers --------------------------

def b64_bytes(data):
    if not data:
        return None
    raw = data.split(",", 1)[1] if "," in data else data
    try:
        return base64.b64decode(raw, validate=False)
    except Exception:
        return None


def decode_color(raw):
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def decode_gray(raw):
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)


def trim_ink(gray):
    if gray is None or gray.size == 0:
        return None
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(bw)
    if pts is None:
        return bw
    x, y, w, h = cv2.boundingRect(pts)
    px = max(4, int(w * 0.06))
    py = max(4, int(h * 0.10))
    return bw[max(0, y - py): min(bw.shape[0], y + h + py),
              max(0, x - px): min(bw.shape[1], x + w + px)]


def image_to_jpeg_bytes(img, quality=82):
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return enc.tobytes() if ok else None


def detect_text_bands(raw):
    """
    원본 한 장만 올렸을 때 대분류/소분류 후보 영역을 자동 분리.
    가장 큰/상단 텍스트 밴드를 대분류, 그 다음 유효 밴드를 소분류로 사용.
    완벽한 OCR 분할이 아니라 Typography profile 생성을 위한 자동 초기값.
    """
    gray = decode_gray(raw)
    if gray is None:
        return None, None
    bw = trim_ink(gray)
    if bw is None:
        return None, None
    # 다시 원본 좌표에서 행 밀도를 사용하기 쉽게 전체 gray 기준 binary.
    _, full = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    density = np.mean(full > 0, axis=1)
    active = density > max(0.012, float(np.percentile(density, 62)) * 0.30)

    bands = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        if (not on or i == len(active)-1) and start is not None:
            end = i if not on else i + 1
            if end - start >= max(8, int(gray.shape[0] * 0.035)):
                bands.append((start, end))
            start = None

    if not bands:
        return raw, None

    # merge close bands
    merged = []
    for y1, y2 in bands:
        if merged and y1 - merged[-1][1] < max(5, int(gray.shape[0] * 0.025)):
            merged[-1] = (merged[-1][0], y2)
        else:
            merged.append((y1, y2))

    candidates = []
    for y1, y2 in merged:
        crop = gray[max(0, y1-4):min(gray.shape[0], y2+4), :]
        t = trim_ink(crop)
        if t is None or t.size == 0:
            continue
        h, w = t.shape[:2]
        ink = float(np.mean(t > 0))
        score = w * h * (0.45 + ink)
        candidates.append((score, y1, y2))

    if not candidates:
        return raw, None

    # Major: largest significant band. Minor: next band spatially different.
    candidates.sort(reverse=True)
    major_band = candidates[0]
    major = gray[max(0, major_band[1]-6):min(gray.shape[0], major_band[2]+6), :]
    major_raw = image_to_jpeg_bytes(major)

    minor_raw = None
    for c in candidates[1:]:
        if abs(c[1] - major_band[1]) > max(10, int(gray.shape[0]*0.05)):
            minor = gray[max(0, c[1]-5):min(gray.shape[0], c[2]+5), :]
            minor_raw = image_to_jpeg_bytes(minor)
            break
    return major_raw, minor_raw


# -------------------------- typography profile --------------------------

def typography_features(raw):
    gray = decode_gray(raw)
    if gray is None:
        return {}
    bw = trim_ink(gray)
    if bw is None or bw.size == 0:
        return {}

    h, w = bw.shape[:2]
    ink = (bw > 0).astype(np.uint8)

    # connected components approximates visual character pieces.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    comps = []
    for i in range(1, n):
        x, y, cw, ch, area = [int(v) for v in stats[i]]
        if area >= max(10, int(w*h*0.0007)) and ch >= max(4, int(h*0.06)):
            comps.append((x, y, cw, ch, area))
    comps.sort(key=lambda q: q[0])

    heights = [c[3] for c in comps] or [h]
    widths = [c[2] for c in comps] or [w]
    mh = float(np.median(heights))

    gaps = []
    for a, b in zip(comps, comps[1:]):
        gaps.append(max(0, b[0] - (a[0] + a[2])) / max(1.0, mh))

    # stroke width proxy
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    pos = dist[dist > 0]
    stroke = 2 * float(np.median(pos)) if len(pos) else 0.0
    stroke_ratio = stroke / max(1.0, mh)

    # stroke variation: serif/calligraphy tends to vary more than gothic.
    skel_like = cv2.Canny(bw, 50, 150)
    edge_density = float(np.mean(skel_like > 0))
    row_var = float(np.std(np.sum(ink, axis=1)) / max(1.0, np.mean(np.sum(ink, axis=1)) + 1e-6))
    col_var = float(np.std(np.sum(ink, axis=0)) / max(1.0, np.mean(np.sum(ink, axis=0)) + 1e-6))

    # slant proxy using fitted line through foreground pixels.
    ys, xs = np.where(ink > 0)
    slant = 0.0
    if len(xs) > 30 and np.std(ys) > 1:
        try:
            slope = np.polyfit(ys.astype(float), xs.astype(float), 1)[0]
            slant = float(np.clip(abs(slope) / max(1.0, w/h), 0, 1))
        except Exception:
            pass

    # Heuristic probabilities. These are "style family likelihoods", not font-name recognition.
    gothic = 0.55 + 0.55 * max(0, 0.14 - row_var) + 0.30 * max(0, 0.12 - slant)
    myeongjo = 0.35 + 0.55 * min(1, row_var) + 0.18 * min(1, edge_density*6)
    gungseo = 0.25 + 0.75 * min(1, slant*1.7 + row_var*0.45 + col_var*0.25)
    handwriting = 0.15 + 0.75 * min(1, slant*1.4 + row_var*0.25 + max(0, 0.08-edge_density)*4)

    vals = np.array([gothic, myeongjo, gungseo, handwriting], dtype=float)
    vals = np.maximum(vals, 0.01)
    vals = vals / vals.sum() * 100.0

    return {
        "aspect_ratio": round(w / max(1.0, h), 3),
        "component_count": len(comps),
        "median_width_to_height": round(float(np.median(widths)) / max(1.0, mh), 3),
        "median_gap_to_height": round(float(np.median(gaps)) if gaps else 0.0, 3),
        "stroke_to_height": round(stroke_ratio, 3),
        "edge_density": round(edge_density, 4),
        "row_variation": round(row_var, 3),
        "column_variation": round(col_var, 3),
        "slant": round(slant, 3),
        "font_family_probabilities": {
            "고딕": round(float(vals[0]), 1),
            "명조": round(float(vals[1]), 1),
            "궁서·붓글씨": round(float(vals[2]), 1),
            "손글씨": round(float(vals[3]), 1),
        },
    }


def typography_similarity(a, b):
    if not a or not b:
        return 0.0

    pa = a.get("font_family_probabilities", {})
    pb = b.get("font_family_probabilities", {})
    keys = ["고딕", "명조", "궁서·붓글씨", "손글씨"]
    va = np.array([float(pa.get(k, 0)) for k in keys])
    vb = np.array([float(pb.get(k, 0)) for k in keys])
    if va.sum() and vb.sum():
        family = 100.0 * (1.0 - np.sum(np.abs(va - vb)) / 200.0)
    else:
        family = 0.0

    def close(key, scale):
        x = float(a.get(key, 0))
        y = float(b.get(key, 0))
        return 100.0 * max(0.0, 1.0 - abs(x-y) / max(scale, abs(x), abs(y), 1e-6))

    geometry = (
        close("aspect_ratio", 1.5) * 0.20
        + close("median_width_to_height", 0.7) * 0.15
        + close("median_gap_to_height", 0.35) * 0.20
        + close("stroke_to_height", 0.12) * 0.25
        + close("slant", 0.25) * 0.20
    )
    return round(family * 0.55 + geometry * 0.45, 1)



# -------------------------- SIFT / Homography Visual Fingerprint --------------------------

def _resize_for_features(gray, max_side=900):
    if gray is None:
        return None, 1.0
    h, w = gray.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return gray, 1.0
    scale = max_side / float(side)
    resized = cv2.resize(gray, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def sift_fingerprint(raw):
    """
    등록 시 1회 계산.
    descriptors는 float32라 JSON 저장용으로 base64+npz 압축.
    keypoint 좌표/크기/각도도 함께 저장.
    """
    gray = decode_gray(raw)
    if gray is None:
        return {}
    gray, scale = _resize_for_features(gray, 900)
    sift = cv2.SIFT_create(nfeatures=1400, contrastThreshold=0.025, edgeThreshold=12, sigma=1.6)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None or not kps:
        return {"count": 0, "scale": scale}

    # Cap descriptors for predictable storage/latency.
    if len(kps) > 900:
        order = sorted(range(len(kps)), key=lambda i: kps[i].response, reverse=True)[:900]
        kps = [kps[i] for i in order]
        desc = desc[order]

    import io
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        descriptors=desc.astype(np.float32),
        points=np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response] for k in kps], dtype=np.float32),
        shape=np.array(gray.shape[:2], dtype=np.int32),
    )
    return {
        "count": int(len(kps)),
        "scale": round(float(scale), 5),
        "npz_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def _load_sift_fingerprint(fp):
    if not fp or not fp.get("npz_b64"):
        return None
    try:
        import io
        raw = base64.b64decode(fp["npz_b64"])
        data = np.load(io.BytesIO(raw))
        return {
            "descriptors": data["descriptors"].astype(np.float32),
            "points": data["points"].astype(np.float32),
            "shape": tuple(int(v) for v in data["shape"]),
        }
    except Exception:
        return None


def sift_homography_score_from_fp(fp, frame_raw):
    ref = _load_sift_fingerprint(fp)
    frame = decode_gray(frame_raw)
    if ref is None or frame is None:
        return {
            "score": 0.0, "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0,
            "median_error": None, "coverage": 0.0, "homography": False
        }

    frame, _ = _resize_for_features(frame, 900)
    sift = cv2.SIFT_create(nfeatures=1600, contrastThreshold=0.025, edgeThreshold=12, sigma=1.6)
    k2, d2 = sift.detectAndCompute(frame, None)
    d1 = ref["descriptors"]
    pts1 = ref["points"]
    if d2 is None or len(d1) < 6 or len(k2) < 6:
        return {
            "score": 0.0, "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0,
            "median_error": None, "coverage": 0.0, "homography": False
        }

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.74 * n.distance:
            good.append(m)

    if len(good) < 4:
        weak = min(24.0, len(good) * 4.0)
        return {
            "score": round(weak, 1), "good_matches": len(good), "inliers": 0,
            "inlier_ratio": 0.0, "median_error": None,
            "coverage": round(min(1.0, len(good)/24.0), 3), "homography": False
        }

    src_pts = np.float32([[pts1[m.queryIdx][0], pts1[m.queryIdx][1]] for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    except cv2.error:
        H, mask = None, None

    if H is None or mask is None:
        return {
            "score": min(32.0, len(good) * 2.0), "good_matches": len(good), "inliers": 0,
            "inlier_ratio": 0.0, "median_error": None,
            "coverage": round(min(1.0, len(good)/35.0), 3), "homography": False
        }

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / max(1, len(good))

    # Reprojection error on inliers.
    projected = cv2.perspectiveTransform(src_pts, H)
    errors = np.linalg.norm(projected.reshape(-1, 2) - dst_pts.reshape(-1, 2), axis=1)
    inlier_errors = errors[inlier_mask] if len(errors) == len(inlier_mask) else errors
    median_error = float(np.median(inlier_errors)) if len(inlier_errors) else 99.0

    # Spatial coverage on the reference image.
    rh, rw = ref["shape"]
    inlier_ref = src_pts.reshape(-1,2)[inlier_mask]
    coverage = 0.0
    if len(inlier_ref) >= 3 and rw > 0 and rh > 0:
        xspan = (float(inlier_ref[:,0].max()) - float(inlier_ref[:,0].min())) / rw
        yspan = (float(inlier_ref[:,1].max()) - float(inlier_ref[:,1].min())) / rh
        coverage = max(0.0, min(1.0, (xspan * yspan) ** 0.5))

    # Score components.
    match_strength = min(100.0, 100.0 * inliers / 45.0)
    ratio_score = min(100.0, inlier_ratio * 100.0)
    error_score = max(0.0, 100.0 * (1.0 - median_error / 8.0))
    coverage_score = min(100.0, coverage * 145.0)

    score = (
        match_strength * 0.34 +
        ratio_score * 0.32 +
        error_score * 0.20 +
        coverage_score * 0.14
    )

    # Strong geometric agreement deserves a floor.
    if inliers >= 18 and inlier_ratio >= 0.70 and median_error <= 3.0:
        score = max(score, 78.0)
    if inliers >= 30 and inlier_ratio >= 0.80 and median_error <= 2.0:
        score = max(score, 88.0)

    return {
        "score": round(min(100.0, score), 1),
        "good_matches": int(len(good)),
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 3),
        "median_error": round(median_error, 2),
        "coverage": round(coverage, 3),
        "homography": True,
    }


def sift_homography_score(ref_raw, frame_raw):
    return sift_homography_score_from_fp(sift_fingerprint(ref_raw), frame_raw)

# -------------------------- full visual / logo --------------------------

def color_profile(raw):
    img = decode_color(raw)
    if img is None:
        return {}
    hsv = cv2.cvtColor(cv2.resize(img, (96, 96)), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    vals = hsv[sat > 35]
    if not len(vals):
        vals = hsv.reshape(-1, 3)
    mean = np.mean(vals, axis=0)
    h = float(mean[0])
    family = (
        "빨강" if h < 10 or h >= 170
        else "주황" if h < 25
        else "노랑" if h < 40
        else "초록" if h < 85
        else "파랑" if h < 130
        else "보라"
    )
    return {
        "family": family,
        "mean_hsv": [round(float(v), 1) for v in mean],
    }


def fit_canvas(bw, W=640, H=260):
    h, w = bw.shape[:2]
    scale = min(W / max(1, w), H / max(1, h))
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    r = cv2.resize(bw, (nw, nh), interpolation=cv2.INTER_AREA)
    c = np.zeros((H, W), np.uint8)
    x, y = (W-nw)//2, (H-nh)//2
    c[y:y+nh, x:x+nw] = r
    return c


def shape_score(ref_raw, frame_raw):
    a = trim_ink(decode_gray(ref_raw))
    b = trim_ink(decode_gray(frame_raw))
    if a is None or b is None:
        return 0.0
    aa, bb = fit_canvas(a), fit_canvas(b)
    corr = max(0.0, float(cv2.matchTemplate(aa, bb, cv2.TM_CCOEFF_NORMED)[0][0]))
    ea, eb = cv2.Canny(aa, 50, 140), cv2.Canny(bb, 50, 140)
    inter = np.logical_and(ea > 0, eb > 0).sum()
    union = np.logical_or(ea > 0, eb > 0).sum()
    iou = float(inter / union) if union else 0.0
    return round(min(100.0, 100.0 * (0.76*corr + 0.24*iou)), 1)


def orb_score(ref_raw, frame_raw):
    a, b = decode_gray(ref_raw), decode_gray(frame_raw)
    if a is None or b is None:
        return 0.0
    orb = cv2.ORB_create(nfeatures=1800, fastThreshold=6)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None or len(k1) < 6 or len(k2) < 6:
        return 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75*n.distance]
    if len(good) < 4:
        return min(30.0, len(good)*6.0)
    coverage = min(1.0, len(good) / max(14.0, min(len(k1), 45.0)))
    inlier = 0.0
    if len(good) >= 6:
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is not None and len(mask):
                inlier = float(mask.sum()) / len(mask)
        except cv2.error:
            pass
    return round(min(100.0, 100.0*(0.55*coverage + 0.45*inlier)), 1)


def visual_score(ref_raw, frame_raw):
    sh = shape_score(ref_raw, frame_raw)
    orb = orb_score(ref_raw, frame_raw)
    return sh, orb, round(max(sh, 0.60*sh + 0.40*orb), 1)


def generate_variants(raw):
    gray = decode_gray(raw)
    if gray is None:
        return {}
    variants = {
        "original": gray,
        "contrast": cv2.convertScaleAbs(gray, alpha=1.45, beta=-18),
        "blur": cv2.GaussianBlur(gray, (5, 5), 0),
        "threshold": cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    }
    out = {}
    for name, img in variants.items():
        b = image_to_jpeg_bytes(img, 78)
        if b:
            out[name] = base64.b64encode(b).decode("ascii")
    return out


def infer_mode(major, minor, major_typ, minor_typ):
    # 기본 등록은 문자 중심. 대분류/소분류가 모두 있으면 TEXT_TYPOGRAPHY.
    # 문자 없는 등록은 LOGO_ONLY.
    if major and minor:
        return "TEXT_TYPOGRAPHY"
    if major:
        return "WORDMARK_OR_TEXT"
    return "LOGO_ONLY"


def build_identity_profile(full_raw, major_raw, minor_raw, major, minor):
    if major_raw is None or (minor and minor_raw is None):
        auto_major, auto_minor = detect_text_bands(full_raw)
        if major_raw is None:
            major_raw = auto_major
        if minor and minor_raw is None:
            minor_raw = auto_minor

    major_typ = typography_features(major_raw or full_raw)
    minor_typ = typography_features(minor_raw) if minor and minor_raw else {}
    return {
        "profile_version": "2.0",
        "mode": infer_mode(major, minor, major_typ, minor_typ),
        "auto_recognition_terms": build_recognition_terms(major, minor),
        "major_typography": major_typ,
        "minor_typography": minor_typ,
        "full_visual": {
            "color": color_profile(full_raw),
            "variants": generate_variants(full_raw),
            "sift": sift_fingerprint(full_raw),
        },
    }


def profile_visual_score(row, frame_raw):
    """
    V0.7.2 FAST:
    1) SIFT+Homography first.
    2) Strong SIFT => return immediately; no ORB/Shape.
    3) Only weak SIFT uses legacy fallback.
    """
    try:
        p = json.loads(row.identity_profile or "{}")
    except Exception:
        p = {}

    sift_fp = p.get("full_visual", {}).get("sift", {})
    if not sift_fp:
        sift_fp = sift_fingerprint(row.reference_image)

    sd = sift_homography_score_from_fp(sift_fp, frame_raw)
    sift_score = float(sd.get("score", 0.0))
    strong = (
        sd.get("homography")
        and sd.get("inliers", 0) >= 20
        and sd.get("inlier_ratio", 0) >= 0.70
        and sift_score >= 88.0
    )

    if strong:
        return round(sift_score, 1), {
            **sd,
            "legacy_shape": None,
            "legacy_orb": None,
            "legacy_visual": None,
            "method": "SIFT_FAST",
        }

    sh, orb, legacy = visual_score(row.reference_image, frame_raw)
    if sd.get("homography") and sd.get("inliers", 0) >= 8:
        final = max(sift_score, sift_score * 0.88 + legacy * 0.12)
        method = "SIFT_HOMOGRAPHY"
    else:
        final = max(sift_score * 0.60 + legacy * 0.40, legacy * 0.75)
        method = "SIFT_FALLBACK"

    return round(min(100.0, final), 1), {
        **sd,
        "legacy_shape": sh,
        "legacy_orb": orb,
        "legacy_visual": legacy,
        "method": method,
    }


# -------------------------- candidate / verification --------------------------

def require_admin():
    return bool(ADMIN_KEY) and request.headers.get("X-Admin-Key", "") == ADMIN_KEY


def gps_ok(row, lat, lon):
    if not row.use_location:
        return True, None
    if lat is None or lon is None or row.latitude is None or row.longitude is None:
        return False, None
    r = 6371000.0
    p1, p2 = math.radians(float(lat)), math.radians(float(row.latitude))
    dp = math.radians(float(row.latitude) - float(lat))
    dl = math.radians(float(row.longitude) - float(lon))
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    d = 2*r*math.asin(math.sqrt(a))
    return d <= row.radius_m, round(d, 1)


def candidate_score(ocr_texts, row):
    f = row.fields()
    major_scores = [(best_term_score(t, f["major_category"]), t) for t in ocr_texts]
    major_score, major_ocr = max(major_scores, default=(0.0, ""))

    minors = split_terms(f["minor_category"])
    minor_hits = []
    for target in minors:
        best = max((best_term_score(t, target), t) for t in ocr_texts) if ocr_texts else (0.0, "")
        minor_hits.append({"target": target, "score": round(best[0], 1), "ocr": best[1]})

    if minors:
        # 하나의 긴 소분류 문자열이 아니라, 등록된 소분류 토큰 중 보이는 증거를 활용.
        good = [h["score"] for h in minor_hits if h["score"] >= 58]
        minor_score = sum(sorted(good, reverse=True)[:3]) / min(3, len(minors)) if good else 0.0
        minor_coverage = len(good) / len(minors)
    else:
        minor_score = 100.0
        minor_coverage = 1.0

    # 대분류는 기준축. 소분류가 존재할 경우 최소 하나 이상의 증거를 요구.
    major_ok = major_score >= 58.0
    minor_ok = (not minors) or any(h["score"] >= 58.0 for h in minor_hits)

    combined = major_score * 0.70 + minor_score * 0.30
    if major_score >= 82:
        combined += 4.0
    return {
        "major_score": round(min(100, major_score), 1),
        "major_ocr": major_ocr,
        "minor_score": round(min(100, minor_score), 1),
        "minor_coverage": round(minor_coverage, 3),
        "minor_hits": minor_hits,
        "text_score": round(min(100, combined), 1),
        "eligible": bool(major_ok and minor_ok),
    }


# -------------------------- routes --------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUPER Registry V0.7.2</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#f5f6f8;color:#17191c;margin:0}
.w{max-width:1220px;margin:22px auto;padding:0 15px}.c{background:#fff;border:1px solid #ddd;border-radius:14px;padding:18px;margin:13px 0}
.g{display:grid;grid-template-columns:1fr 1fr;gap:12px}.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
label{display:block;font-size:13px;color:#555;margin-bottom:5px}input,select{width:100%;box-sizing:border-box;padding:10px;border:1px solid #ccd1d7;border-radius:8px}
.full{grid-column:1/-1}.row{display:flex;gap:8px;align-items:center}.row input[type=checkbox]{width:auto}
button{border:0;border-radius:8px;background:#111;color:#fff;font-weight:700;padding:10px 13px;cursor:pointer}
.muted{font-size:13px;color:#666}.ok{color:#18723a;font-weight:700}.warn{color:#a22;font-weight:700}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
.profile{white-space:pre-wrap;font:12px/1.5 ui-monospace,monospace;background:#101114;color:#eee;padding:10px;border-radius:8px}
@media(max-width:820px){.g,.g3{grid-template-columns:1fr}.full{grid-column:auto}table{display:block;overflow:auto}}
</style></head>
<body><div class="w">
<h2>LUPER LIVE Registry V0.7.2</h2>
<p><b>문자 → 후보 → Typography → SIFT+Homography Visual ID 확정 → GPS → 5초 링크카드</b></p>

<div class="c">
<label>관리자 키</label><div class="row"><input id="admin" type="password"><button id="saveKey">키 저장</button></div>
</div>

<div class="c"><form id="f"><div class="g">
<div><label>등록명</label><input id="regname" required placeholder="살만온족발_메뉴_01"></div>
<div><label>표시명</label><input id="display" required placeholder="살만온족발 메뉴"></div>
<div><label>대분류 · 기준 문자/상호</label><input id="major" required placeholder="살만온족발"></div>
<div><label>소분류 · 보조 문자</label><input id="minor" placeholder="족발 보쌈 닭발"></div>
<div class="full"><label>인식문자 · 자동 생성</label><input id="recognition" disabled></div>
<div class="full"><label>URL</label><input id="url" required placeholder="https://..."></div>
<div class="full"><label>전체 기준 이미지</label><input id="fullImg" type="file" accept="image/*" required></div>
<div><label>대분류 문자 Crop (선택 · 없으면 자동추정)</label><input id="majorImg" type="file" accept="image/*"></div>
<div><label>소분류 문자 Crop (선택 · 없으면 자동추정)</label><input id="minorImg" type="file" accept="image/*"></div>
<div><label>등록 유형</label><select id="mode"><option value="auto">자동</option><option value="TEXT_TYPOGRAPHY">문자+글자체 중심</option><option value="WORDMARK_OR_TEXT">워드마크/문자</option><option value="TEXT_LOGO">문자+로고</option><option value="LOGO_ONLY">로고만</option></select></div>
<div><label>기본 Visual 기준</label><input id="thr" type="number" value="48" min="0" max="100"></div>
<div class="row"><input id="useGps" type="checkbox"><label for="useGps" style="margin:0">GPS 사용</label></div>
<div><label>반경(m)</label><input id="radius" type="number" value="150"></div>
<div><label>위도</label><input id="lat" type="number" step="any"></div><div><label>경도</label><input id="lon" type="number" step="any"></div>
<div class="full"><button>등록 + 원본 사전분석</button></div>
</div></form></div>

<div class="c">
<h3>등록목록 / 사전분석 결과</h3><div id="state" class="muted"></div><div id="list"></div>
</div>

<div class="c">
<h3>최근 LIVE TRACE</h3>
<p class="muted">이미지는 저장하지 않고 OCR·후보·점수·링크카드 결과만 기록합니다.</p>
<button type="button" onclick="loadTraces()">최근 50건 불러오기</button>
<div id="traces" class="profile" style="margin-top:10px">TRACE 대기</div>
</div>
</div>
<script>
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
$('admin').value=sessionStorage.getItem('luper_admin')||'';
$('saveKey').onclick=()=>{sessionStorage.setItem('luper_admin',$('admin').value);refresh()};
const ah=()=>({'X-Admin-Key':$('admin').value});
function file64(f){return new Promise((ok,no)=>{if(!f)return ok(null);let r=new FileReader();r.onload=()=>ok(r.result);r.onerror=no;r.readAsDataURL(f)})}
function updateRec(){const a=$('major').value.trim(), b=$('minor').value.trim();const ms=b.split(/[\s,|/·]+/).filter(Boolean);let t=[];if(a)t.push(a);for(const m of ms){t.push(m);if(a){t.push(a+' '+m);t.push(a+m)}}$('recognition').value=[...new Set(t)].join(' | ')}
$('major').oninput=updateRec;$('minor').oninput=updateRec;

async function refresh(){
 const h=await (await fetch('/health')).json();$('state').innerHTML=`<span class="ok">ONLINE</span> V${h.version} · ${h.entries}개`;
 const r=await fetch('/api/entries',{headers:ah()});if(r.status===401){$('list').innerHTML='<p class="warn">관리자 키를 저장하세요.</p>';return}
 const a=await r.json();let s='<table><tr><th>등록명</th><th>대분류</th><th>소분류</th><th>자동 인식문자</th><th>타이포그래피</th><th>유형</th><th>URL</th><th></th></tr>';
 for(const x of a){
  const M=x.profile_summary?.major_typography||{}, m=x.profile_summary?.minor_typography||{};
  s+=`<tr><td><b>${esc(x.registration_name)}</b></td><td>${esc(x.major_category)}</td><td>${esc(x.minor_category)}</td><td>${esc(x.recognition_text)}</td><td>대: ${esc(JSON.stringify(M))}<br>소: ${esc(JSON.stringify(m))}</td><td>${esc(x.profile_summary?.mode||x.match_mode)}</td><td><a href="${esc(x.url)}" target="_blank">열기</a></td><td><button onclick="delx(${x.id})">삭제</button></td></tr>`;
 }
 $('list').innerHTML=s+'</table>';
}
async function delx(id){if(!confirm('삭제할까요?'))return;await fetch('/api/entries/'+id,{method:'DELETE',headers:ah()});refresh()}
$('f').onsubmit=async e=>{
 e.preventDefault();if(!$('admin').value)return alert('관리자 키를 저장하세요.');
 const body={registration_name:$('regname').value,display_name:$('display').value,major_category:$('major').value,minor_category:$('minor').value,url:$('url').value,match_mode:$('mode').value,visual_threshold:+$('thr').value||48,use_location:$('useGps').checked,radius_m:+$('radius').value||150,latitude:$('lat').value?+$('lat').value:null,longitude:$('lon').value?+$('lon').value:null,
 reference_image_base64:await file64($('fullImg').files[0]),major_image_base64:await file64($('majorImg').files[0]),minor_image_base64:await file64($('minorImg').files[0])};
 const r=await fetch('/api/entries',{method:'POST',headers:{'Content-Type':'application/json','X-Admin-Key':$('admin').value},body:JSON.stringify(body)});
 const x=await r.json();if(!r.ok)return alert(JSON.stringify(x));
 alert('등록 및 사전분석 완료');refresh();
};

async function loadTraces(){
 const r=await fetch('/api/traces?limit=50',{headers:ah()});
 if(!r.ok){$('traces').textContent='TRACE 조회 실패';return}
 const a=await r.json();
 $('traces').textContent=a.map(x=>`${x.created_at||''} | ${x.session_id} | ${x.event}\n${JSON.stringify(x.payload)}`).join('\n\n');
}

refresh();updateRec();
</script></body></html>"""


@app.get("/")
def index():
    return Response(INDEX_HTML, content_type="text/html; charset=utf-8")


@app.get("/health")
def health():
    return jsonify(ok=True, version=VERSION, database=db.engine.url.get_backend_name(), entries=Link.query.count(), traces=LiveTrace.query.count())


@app.get("/api/entries")
def entries():
    if not require_admin():
        return jsonify(error="unauthorized"), 401
    rows = Link.query.filter_by(enabled=True).order_by(Link.priority.desc(), Link.id.desc()).all()
    return jsonify([r.public_dict() for r in rows])


@app.post("/api/entries")
def create_entry():
    if not require_admin():
        return jsonify(error="unauthorized"), 401
    x = request.get_json(silent=True) or {}
    major = str(x.get("major_category", "")).strip()
    minor = str(x.get("minor_category", "")).strip()
    if not all([x.get("registration_name"), x.get("display_name"), major, x.get("url"), x.get("reference_image_base64")]):
        return jsonify(error="등록명/표시명/대분류/URL/전체 기준이미지는 필수입니다."), 400

    full_raw = b64_bytes(x.get("reference_image_base64"))
    major_raw = b64_bytes(x.get("major_image_base64"))
    minor_raw = b64_bytes(x.get("minor_image_base64"))
    if decode_gray(full_raw) is None:
        return jsonify(error="전체 기준이미지를 읽을 수 없습니다."), 400

    profile = build_identity_profile(full_raw, major_raw, minor_raw, major, minor)
    mode = str(x.get("match_mode") or "auto")
    if mode == "auto":
        mode = profile.get("mode") or "TEXT_TYPOGRAPHY"
    profile["mode"] = mode

    item = Link(
        key_text=major,
        registration_name=str(x["registration_name"]).strip(),
        major_category=major,
        minor_category=minor,
        recognition_text=build_recognition_text(major, minor),
        display_name=str(x["display_name"]).strip(),
        url=str(x["url"]).strip(),
        match_mode=mode,
        reference_image=full_raw,
        major_reference=major_raw,
        minor_reference=minor_raw,
        visual_threshold=float(x.get("visual_threshold") or 48),
        priority=int(x.get("priority") or 0),
        use_location=bool(x.get("use_location")),
        latitude=float(x["latitude"]) if x.get("latitude") is not None else None,
        longitude=float(x["longitude"]) if x.get("longitude") is not None else None,
        radius_m=float(x.get("radius_m") or 150),
        identity_profile=json.dumps(profile, ensure_ascii=False),
    )
    db.session.add(item)
    db.session.commit()
    return jsonify(item.public_dict()), 201


@app.delete("/api/entries/<int:item_id>")
def delete_entry(item_id):
    if not require_admin():
        return jsonify(error="unauthorized"), 401
    item = db.session.get(Link, item_id)
    if not item:
        return jsonify(error="not found"), 404
    db.session.delete(item)
    db.session.commit()
    return jsonify(ok=True)


@app.get("/api/index")
def registry_index():
    rows=Link.query.filter_by(enabled=True).order_by(Link.priority.desc(),Link.id.desc()).all()
    data=[]
    for r in rows:
        f=r.fields()
        data.append({"id":r.id,"major_category":f["major_category"],"minor_category":f["minor_category"],"recognition_terms":build_recognition_terms(f["major_category"],f["minor_category"]),"priority":r.priority})
    return jsonify(entries=data,version=VERSION)

@app.post("/api/candidates")
def candidates():
    x = request.get_json(silent=True) or {}
    texts = [str(t).strip() for t in (x.get("texts") or []) if len(norm(str(t))) >= 2]
    if not texts:
        return jsonify(candidates=[], version=VERSION)

    out = []
    for row in Link.query.filter_by(enabled=True).all():
        c = candidate_score(texts, row)
        if c["eligible"]:
            d = row.public_dict()
            d.update(c)
            out.append(d)
    out.sort(key=lambda q: (q["text_score"], q["priority"]), reverse=True)
    return jsonify(candidates=out[:8], version=VERSION)


@app.post("/api/verify")
def verify():
    x = request.get_json(silent=True) or {}
    lat, lon = x.get("lat"), x.get("lon")
    candidate_ids = [int(v) for v in (x.get("candidate_ids") or [])[:3]]
    crops = x.get("crops") or []

    crop_items = []
    for c in crops[:3]:
        raw = b64_bytes((c or {}).get("image_base64"))
        text = str((c or {}).get("text", "")).strip()
        if raw and decode_gray(raw) is not None:
            crop_items.append((text, raw, int((c or {}).get("frame_index", 0))))

    results, diagnostics = [], []
    for cid in candidate_ids:
        row = db.session.get(Link, cid)
        if not row or not row.enabled:
            continue
        f = row.fields()
        try:
            profile = json.loads(row.identity_profile or "{}")
        except Exception:
            profile = {}

        best_major = (0.0, "", None)
        best_minor = (0.0, "", None)
        best_visual = (0.0, {})
        frame_visuals = {}

        # Typography compares only candidate-relevant crops.
        for txt, raw, frame_index in crop_items:
            maj_text = best_term_score(txt, f["major_category"])
            if maj_text >= 48:
                typ = typography_similarity(profile.get("major_typography", {}), typography_features(raw))
                if typ > best_major[0]:
                    best_major = (typ, txt, raw)

            if f["minor_category"]:
                minor_text = max((best_term_score(txt, t) for t in split_terms(f["minor_category"])), default=0)
                if minor_text >= 48:
                    typ = typography_similarity(profile.get("minor_typography", {}), typography_features(raw))
                    if typ > best_minor[0]:
                        best_minor = (typ, txt, raw)

            vs, vdiag = profile_visual_score(row, raw)
            if vs > best_visual[0]:
                best_visual = (vs, vdiag)
            prev = frame_visuals.get(frame_index)
            if prev is None or vs > prev[0]:
                frame_visuals[frame_index] = (vs, vdiag)

            # One very strong geometric identification is enough for a fast path.
            if (
                vdiag.get("method") == "SIFT_FAST"
                and vs >= 92.0
                and vdiag.get("inliers", 0) >= 30
                and vdiag.get("inlier_ratio", 0) >= 0.80
            ):
                break

        # Full registration mode controls visual weight / floor.
        mode = row.match_mode or profile.get("mode") or "TEXT_TYPOGRAPHY"
        major_typ = best_major[0]
        minor_typ = best_minor[0] if f["minor_category"] else 100.0
        visual_values = sorted((v[0] for v in frame_visuals.values()), reverse=True)
        best_diag = best_visual[1] if len(best_visual) > 1 else {}
        if (
            visual_values
            and best_diag.get("method") == "SIFT_FAST"
            and best_diag.get("inliers", 0) >= 30
            and best_diag.get("inlier_ratio", 0) >= 0.80
        ):
            visual = visual_values[0]
        elif len(visual_values) >= 2:
            visual = sum(visual_values[:2]) / 2.0
        elif visual_values:
            visual = visual_values[0]
        else:
            visual = 0.0

        # Recompute text evidence from crop texts for final score.
        tc = candidate_score([t for t, _, _ in crop_items], row)
        text_score = tc["text_score"]

        if mode == "LOGO_ONLY":
            weights = (0.00, 0.00, 1.00)
            visual_floor = max(68.0, row.visual_threshold)
        elif mode == "WORDMARK_OR_TEXT":
            weights = (0.55, 0.20, 0.25)
            visual_floor = max(48.0, row.visual_threshold - 4)
        elif mode == "TEXT_LOGO":
            weights = (0.58, 0.22, 0.20)
            visual_floor = max(42.0, row.visual_threshold - 6)
        else:  # TEXT_TYPOGRAPHY
            weights = (0.62, 0.28, 0.10)
            visual_floor = max(36.0, row.visual_threshold - 10)

        typo_score = major_typ * 0.70 + minor_typ * 0.30
        final_conf = text_score * weights[0] + typo_score * weights[1] + visual * weights[2]

        # Strong text + typography is allowed to survive a low logo/full-image score.
        visual_ok = visual >= visual_floor
        if mode != "LOGO_ONLY" and text_score >= 84 and typo_score >= 58:
            strong_sift = (
            best_visual[1].get("method") == "SIFT_HOMOGRAPHY"
            and best_visual[1].get("score", 0) >= 90.0
            and best_visual[1].get("inliers", 0) >= 20
            and best_visual[1].get("inlier_ratio", 0) >= 0.70
        )
        visual_ok = visual >= max(30.0, visual_floor - 10)
        if strong_sift:
            visual_ok = True

        gok, dist = gps_ok(row, lat, lon)
        passed = bool(tc["eligible"] and visual_ok and gok and final_conf >= (58 if mode != "LOGO_ONLY" else 66))

        d = {
            "id": row.id,
            "registration_name": f["registration_name"],
            "display_name": f["display_name"],
            "major_category": f["major_category"],
            "minor_category": f["minor_category"],
            "url": row.url,
            "match_mode": mode,
            "text_score": round(text_score, 1),
            "major_text_score": tc["major_score"],
            "minor_text_score": tc["minor_score"],
            "major_typography_score": round(major_typ, 1),
            "minor_typography_score": round(minor_typ, 1),
            "typography_score": round(typo_score, 1),
            "visual_score": round(visual, 1),
            "visual_floor": round(visual_floor, 1),
            "visual_method": best_visual[1].get("method", "none") if len(best_visual) > 1 else "none",
            "sift_good_matches": best_visual[1].get("good_matches", 0) if len(best_visual) > 1 else 0,
            "sift_inliers": best_visual[1].get("inliers", 0) if len(best_visual) > 1 else 0,
            "sift_inlier_ratio": best_visual[1].get("inlier_ratio", 0) if len(best_visual) > 1 else 0,
            "sift_median_error": best_visual[1].get("median_error") if len(best_visual) > 1 else None,
            "sift_coverage": best_visual[1].get("coverage", 0) if len(best_visual) > 1 else 0,
            "legacy_shape": best_visual[1].get("legacy_shape", 0) if len(best_visual) > 1 else 0,
            "legacy_orb": best_visual[1].get("legacy_orb", 0) if len(best_visual) > 1 else 0,
            "visual_frames": len(frame_visuals),
            "strong_sift": strong_sift,
            "fast_path": fast_path,
            "final_confidence": round(final_conf, 1),
            "gps_ok": gok,
            "distance_m": dist,
            "passed": passed,
        }
        diagnostics.append(d)
        if passed:
            results.append(d)

    results.sort(key=lambda q: (q["final_confidence"], q["text_score"]), reverse=True)
    diagnostics.sort(key=lambda q: (q["passed"], q["final_confidence"]), reverse=True)
    return jsonify(matches=results, diagnostics=diagnostics, version=VERSION)



@app.post("/api/trace")
def trace_event():
    x = request.get_json(silent=True) or {}
    sid = str(x.get("session_id", "")).strip()
    event = str(x.get("event", "")).strip()
    payload = x.get("payload") or {}
    if not sid or not event:
        return jsonify(error="session_id/event required"), 400
    # raw images are deliberately not accepted/stored.
    item = LiveTrace(
        session_id=sid[:80],
        event=event[:80],
        payload=json.dumps(payload, ensure_ascii=False)[:12000],
    )
    db.session.add(item)
    db.session.commit()
    return jsonify(ok=True)


@app.get("/api/traces")
def traces():
    if not require_admin():
        return jsonify(error="unauthorized"), 401
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    rows = LiveTrace.query.order_by(LiveTrace.id.desc()).limit(limit).all()
    return jsonify([r.public_dict() for r in rows])


def migrate():
    db.create_all()
    insp = inspect(db.engine)
    if "links" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("links")}
    additions = [
        ("registration_name", "VARCHAR(300)"),
        ("major_category", "VARCHAR(300)"),
        ("minor_category", "VARCHAR(500)"),
        ("recognition_text", "VARCHAR(1500)"),
        ("match_mode", "VARCHAR(30)"),
        ("major_reference", "BLOB"),
        ("minor_reference", "BLOB"),
        ("identity_profile", "TEXT"),
    ]
    with db.engine.begin() as conn:
        for name, typ in additions:
            if name not in cols:
                conn.execute(sql_text(f"ALTER TABLE links ADD COLUMN {name} {typ}"))
        conn.execute(sql_text("""
        UPDATE links SET
          registration_name=COALESCE(NULLIF(registration_name,''),display_name,key_text),
          major_category=COALESCE(NULLIF(major_category,''),key_text),
          minor_category=COALESCE(minor_category,''),
          recognition_text=COALESCE(NULLIF(recognition_text,''),key_text),
          match_mode=COALESCE(NULLIF(match_mode,''),'auto')
        """))

    # legacy rows receive the new profile.
    for row in Link.query.all():
        f = row.fields()
        desired = build_recognition_text(f["major_category"], f["minor_category"])
        changed = False
        if row.recognition_text != desired:
            row.recognition_text = desired
            changed = True
        try:
            current = json.loads(row.identity_profile or "{}")
        except Exception:
            current = {}
        needs_rebuild = (
            not current or
            not current.get("full_visual", {}).get("sift", {}).get("npz_b64")
        )
        if needs_rebuild:
            try:
                p = build_identity_profile(
                    row.reference_image,
                    row.major_reference,
                    row.minor_reference,
                    f["major_category"],
                    f["minor_category"],
                )
                if row.match_mode and row.match_mode != "auto":
                    p["mode"] = row.match_mode
                row.identity_profile = json.dumps(p, ensure_ascii=False)
                changed = True
            except Exception:
                pass
        if changed:
            db.session.add(row)
    db.session.commit()


with app.app_context():
    migrate()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8788")), debug=False)
