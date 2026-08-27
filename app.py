import base64
import time
from functools import lru_cache
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
VERSION = "0.8.1.3"


class Link(db.Model):
    __tablename__ = "links"

    id = db.Column(db.Integer, primary_key=True)
    key_text = db.Column(db.String(300), nullable=False, default="")
    registration_name = db.Column(db.String(300))
    major_category = db.Column(db.String(300))
    minor_category = db.Column(db.String(500))
    recognition_text = db.Column(db.String(1500))
    display_name = db.Column(db.String(300), nullable=False)
    group_name = db.Column(db.String(300), nullable=False, default="")
    action_name = db.Column(db.String(300), nullable=False, default="")
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
        registration=(self.registration_name or self.display_name or self.key_text or "").strip()
        display=(self.display_name or registration).strip()
        try:p=json.loads(self.identity_profile or "{}")
        except Exception:p={}
        hints=p.get("context",{}).get("recognition_terms") or context_recognition_terms(registration,display)
        return {
            "registration_name":registration,
            "major_category":(self.major_category or "").strip(),
            "minor_category":(self.minor_category or "").strip(),
            "recognition_text":" | ".join(hints),
            "display_name":display,
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
            "group_name": (self.group_name or f["registration_name"]).strip(),
            "action_name": (self.action_name or "").strip(),
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
                "hint_tokens": profile.get("context", {}).get("hint_tokens", []),
                "image_ocr_terms": profile.get("context", {}).get("image_ocr_terms", []),
                "recognition_terms": profile.get("context", {}).get("recognition_terms", []),
                "color": profile.get("full_visual", {}).get("color", {}),
                "design": profile.get("design", {}),
                "text_count": len(profile.get("text_identity", [])),
                "sift_features": profile.get("full_visual", {}).get("sift", {}).get("count", 0),
                "ocr_ready": bool(profile.get("text_identity", [])),
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



def context_tokens(registration_name, display_name):
    generic={"로고","logo","이미지","image","사진","전면","후면","정면","메뉴","링크","공식","사이트","홈페이지","사용설명서","설명서","as","서비스","페이지"}
    raw=f"{registration_name or ''} {display_name or ''}"
    parts=re.findall(r"[가-힣A-Za-z]+|\d{2,}",raw)
    out=[]
    for p in parts:
        q=p.strip()
        if not q or q.lower() in generic: continue
        if len(q)==1 and not q.isdigit(): continue
        out.append(q)
    return list(dict.fromkeys(out))

def context_recognition_terms(registration_name, display_name):
    toks=context_tokens(registration_name,display_name)
    terms=list(toks)
    for i in range(len(toks)-1):
        terms += [toks[i]+toks[i+1], toks[i]+" "+toks[i+1]]
    if len(toks)>=2: terms += [" ".join(toks), "".join(toks)]
    return list(dict.fromkeys(terms))

def region_profile(raw):
    if not raw:return {}
    return {"typography":typography_features(raw),"color":color_profile(raw),"sift":sift_fingerprint(raw)}


def _safe_bbox(item):
    b=(item or {}).get("bbox") or {}
    try:
        x0=max(0,int(b.get("x0",0))); y0=max(0,int(b.get("y0",0)))
        x1=max(x0+1,int(b.get("x1",0))); y1=max(y0+1,int(b.get("y1",0)))
        return x0,y0,x1,y1
    except Exception:
        return None

def normalize_detected_texts(items):
    out=[]
    for item in items or []:
        text=str((item or {}).get("text","")).strip()
        if len(norm(text))<2: continue
        box=_safe_bbox(item)
        if not box: continue
        out.append({
            "text":text,
            "confidence":round(float((item or {}).get("confidence",0) or 0),1),
            "bbox":{"x0":box[0],"y0":box[1],"x1":box[2],"y1":box[3]},
        })
    return out[:80]



TEXT_LIKENESS_MIN_SCORE=58.0

def _char_quality(text):
    chars=[c for c in str(text) if not c.isspace()]
    if not chars:
        return {"useful_ratio":0.0,"alpha_ratio":0.0,"symbol_ratio":1.0,"repeat_ratio":1.0}
    useful=sum(1 for c in chars if c.isalnum() or ("가"<=c<="힣"))
    alpha=sum(1 for c in chars if c.isalpha() or ("가"<=c<="힣"))
    symbols=len(chars)-useful
    maxrun=1
    run=1
    for i in range(1,len(chars)):
        if chars[i]==chars[i-1]:
            run+=1
            maxrun=max(maxrun,run)
        else:
            run=1
    return {
        "useful_ratio":useful/max(1,len(chars)),
        "alpha_ratio":alpha/max(1,len(chars)),
        "symbol_ratio":symbols/max(1,len(chars)),
        "repeat_ratio":maxrun/max(1,len(chars)),
    }

def _crop_text_likeness(raw,bbox):
    crop=crop_raw_bbox(raw,bbox,2)
    if not crop:
        return {"score":0.0}
    img=decode_gray(crop)
    if img is None or img.size==0:
        return {"score":0.0}
    h,w=img.shape[:2]
    if h<6 or w<8:
        return {"score":0.0}

    target_h=80
    scale=target_h/max(1,h)
    nw=max(8,int(w*scale))
    img=cv2.resize(img,(nw,target_h),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)

    blur=cv2.GaussianBlur(img,(3,3),0)
    _,bw1=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _,bw2=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    def measure(bw):
        total=bw.size
        ink=float(np.count_nonzero(bw))/max(1,total)
        n,labels,stats,_=cv2.connectedComponentsWithStats(bw,8)
        comps=[]
        for i in range(1,n):
            x,y,cw,ch,area=stats[i]
            if area<6 or ch<4 or cw<2:
                continue
            if area>total*0.50:
                continue
            comps.append((x,y,cw,ch,area))

        comp_count=len(comps)
        if comp_count:
            hs=[c[3] for c in comps]
            med_h=float(np.median(hs))
            h_consistency=float(np.mean([1 for hh in hs if 0.45*med_h<=hh<=1.9*med_h]))
            aspect_ok=float(np.mean([1 for _,_,cw,ch,_ in comps if 0.08<=cw/max(1,ch)<=3.8]))
        else:
            h_consistency=0.0
            aspect_ok=0.0

        edges=cv2.Canny(img,50,130)
        edge_density=float(np.mean(edges>0))
        ink_score=max(0.0,1.0-abs(ink-0.24)/0.24)
        comp_score=min(1.0,comp_count/8.0)
        edge_score=min(1.0,edge_density/0.18)
        score=100.0*(0.28*ink_score+0.28*comp_score+0.18*h_consistency+0.14*aspect_ok+0.12*edge_score)
        return {
            "score":score,
            "ink_ratio":ink,
            "component_count":comp_count,
            "height_consistency":h_consistency,
            "aspect_ok":aspect_ok,
            "edge_density":edge_density,
        }

    a=measure(bw1)
    b=measure(bw2)
    return a if a["score"]>=b["score"] else b

def apply_text_likeness_gate(items,full_raw,registration_name="",display_name=""):
    rows=normalize_detected_texts(items)
    out=[]
    for r in rows:
        b=r["bbox"]
        bbox=(b["x0"],b["y0"],b["x1"],b["y1"])
        charq=_char_quality(r["text"])
        visual=_crop_text_likeness(full_raw,bbox)
        hint_score=_hint_similarity_simple(r["text"],registration_name,display_name)

        conf=float(r.get("confidence",0) or 0)
        useful=charq["useful_ratio"]
        symbols=charq["symbol_ratio"]
        repeat=charq["repeat_ratio"]
        visual_score=float(visual.get("score",0) or 0)

        score=(visual_score*0.48+min(100.0,conf)*0.22+useful*100.0*0.20+min(100.0,hint_score)*0.10)
        if symbols>0.42: score-=18
        if repeat>0.34: score-=14
        if len(norm(r["text"]))<2: score-=25

        passed=score>=TEXT_LIKENESS_MIN_SCORE
        if conf>=82 and hint_score>=78:
            passed=True

        reason="TEXT_LIKE" if passed else "NON_TEXT_TEXTURE"
        if symbols>0.42 and not passed:
            reason="SYMBOL_HEAVY"
        elif visual_score<35 and not passed:
            reason="LOW_TEXT_STRUCTURE"

        x=dict(r)
        x["text_likeness_score"]=round(max(0.0,min(100.0,score)),1)
        x["text_likeness_visual"]=round(visual_score,1)
        x["char_useful_ratio"]=round(useful*100,1)
        x["text_likeness_passed"]=bool(passed)
        x["text_likeness_reason"]=reason
        x["hint_score"]=round(hint_score,1)
        out.append(x)
    return out

TEXT_SIZE_GATE_PCT=5.0
TEXT_SIZE_SOFT_MIN_PCT=3.0
TEXT_SIZE_SOFT_CONF=75.0

def _hint_similarity_simple(text,registration_name,display_name):
    target_terms=context_recognition_terms(registration_name,display_name)
    if not target_terms:return 0.0
    return max((best_term_score(text,t) for t in target_terms),default=0.0)

def apply_text_size_gate(items,image_h,registration_name="",display_name=""):
    rows=normalize_detected_texts(items)
    out=[]
    for r in rows:
        b=r["bbox"]
        hpct=(max(1,b["y1"]-b["y0"])/max(1,image_h))*100.0
        conf=float(r.get("confidence",0) or 0)
        hint_score=_hint_similarity_simple(r["text"],registration_name,display_name)

        passed=False
        reason=""
        if hpct>=TEXT_SIZE_GATE_PCT:
            passed=True
            reason="HEIGHT>=5%"
        elif hpct>=TEXT_SIZE_SOFT_MIN_PCT and (conf>=TEXT_SIZE_SOFT_CONF or hint_score>=72.0):
            passed=True
            reason="HEIGHT_3~5%_SOFT_PASS"
        elif hpct<TEXT_SIZE_SOFT_MIN_PCT:
            passed=False
            reason="HEIGHT<3%"
        else:
            passed=False
            reason="HEIGHT_3~5%_LOW_CONF"

        x=dict(r)
        x["height_pct"]=round(hpct,1)
        x["hint_score"]=round(hint_score,1)
        x["size_gate_passed"]=passed
        x["size_gate_reason"]=reason
        out.append(x)
    return out

def detected_text_terms(items,image_h=None,registration_name="",display_name=""):
    rows=normalize_detected_texts(items)
    if image_h is not None:
        rows=[r for r in rows if r.get("size_gate_passed",True) and r.get("text_likeness_passed",True)]

    def usable(r):
        t=r["text"].strip()
        chars=[c for c in t if not c.isspace()]
        if not chars:return False
        useful=sum(1 for c in chars if c.isalnum() or ("가"<=c<="힣"))
        ratio=useful/max(1,len(chars))
        return r.get("confidence",0)>=42 and ratio>=0.65

    rows=[r for r in rows if usable(r)]
    terms=[r["text"] for r in rows if r["text"]]
    ordered=sorted(rows,key=lambda r:(r["bbox"]["y0"],r["bbox"]["x0"]))
    for a,b in zip(ordered,ordered[1:]):
        ah=max(1,a["bbox"]["y1"]-a["bbox"]["y0"])
        bh=max(1,b["bbox"]["y1"]-b["bbox"]["y0"])
        gap=b["bbox"]["y0"]-a["bbox"]["y1"]
        overlap=max(0,min(a["bbox"]["x1"],b["bbox"]["x1"])-max(a["bbox"]["x0"],b["bbox"]["x0"]))
        minw=max(1,min(a["bbox"]["x1"]-a["bbox"]["x0"],b["bbox"]["x1"]-b["bbox"]["x0"]))
        if gap<=max(ah,bh)*1.2 and overlap/minw>=0.25:
            terms.extend([a["text"]+" "+b["text"],a["text"]+b["text"]])
    return list(dict.fromkeys(t for t in terms if len(norm(t))>=2))

def script_type(text):
    h=sum(1 for c in text if "가"<=c<="힣")
    e=sum(1 for c in text if "a"<=c.lower()<="z")
    n=sum(1 for c in text if c.isdigit())
    if h and e:return "한글+영문"
    if h:return "한글"
    if e:return "영문"
    if n:return "숫자"
    return "기타"

def crop_raw_bbox(raw,bbox,pad=4):
    img=decode_color(raw)
    if img is None:return None
    h,w=img.shape[:2]
    x0,y0,x1,y1=bbox
    x0=max(0,x0-pad); y0=max(0,y0-pad); x1=min(w,x1+pad); y1=min(h,y1+pad)
    if x1<=x0 or y1<=y0:return None
    crop=img[y0:y1,x0:x1]
    ok,enc=cv2.imencode(".jpg",crop,[cv2.IMWRITE_JPEG_QUALITY,88])
    return enc.tobytes() if ok else None

def palette_profile(raw):
    img=decode_color(raw)
    if img is None:return {}
    img=cv2.resize(img,(120,120),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H,S,V=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    chroma=(S>40)&(V>=58)
    red=chroma&((H<10)|(H>=170))
    masks={
      "화이트":(V>=205)&(S<=45),
      "블랙":V<58,
      "회색":(V>=58)&(V<205)&(S<=40),
      "버건디":red&(V<145),
      "레드":red&(V>=145),
      "주황":chroma&(H>=10)&(H<25),
      "노랑":chroma&(H>=25)&(H<40),
      "초록":chroma&(H>=40)&(H<85),
      "파랑":chroma&(H>=85)&(H<130),
      "보라":chroma&(H>=130)&(H<170),
    }
    total=float(img.shape[0]*img.shape[1])
    items=[]
    for name,mask in masks.items():
        pct=float(np.count_nonzero(mask))/total*100.0
        if pct>=0.2: items.append({"family":name,"percent":round(pct,1)})
    items.sort(key=lambda x:x["percent"],reverse=True)
    return {"dominant":items[:6]}

def region_identity_profiles(full_raw,detected):
    img=decode_color(full_raw)
    if img is None:return []
    H,W=img.shape[:2]
    out=[]
    gated=normalize_detected_texts(detected)
    meta={}
    for rr in detected or []:
        try:
            bb=rr["bbox"]
            meta[(rr["text"],bb["x0"],bb["y0"],bb["x1"],bb["y1"])]=rr
        except Exception:
            pass
    for r in gated:
        b=r["bbox"];box=(b["x0"],b["y0"],b["x1"],b["y1"])
        key=(r["text"],b["x0"],b["y0"],b["x1"],b["y1"])
        if key in meta:r.update(meta[key])
        cw=max(1,box[2]-box[0]);ch=max(1,box[3]-box[1])
        wr=cw/max(1,W);hr=ch/max(1,H);area=wr*hr
        crop=crop_raw_bbox(full_raw,box,4)
        typ=typography_features(crop) if crop else {}
        col=palette_profile(crop) if crop else {}
        importance=min(100.0,hr*540+wr*28+area*180+r["confidence"]*0.10)
        valid_text=bool(r.get("text_likeness_passed",True) and r.get("size_gate_passed",True))
        role=("PRIMARY" if importance>=55 else ("SECONDARY" if importance>=27 else "DETAIL")) if valid_text else "EXCLUDED"
        fam=typ.get("font_family_probabilities",{})
        top_font=max(fam.items(),key=lambda kv:kv[1]) if fam else None
        out.append({
          "text":r["text"],"script":script_type(r["text"]),"confidence":r["confidence"],
          "height_pct":r.get("height_pct"),"size_gate_passed":r.get("size_gate_passed"),"size_gate_reason":r.get("size_gate_reason"),
          "text_likeness_score":r.get("text_likeness_score"),"text_likeness_visual":r.get("text_likeness_visual"),
          "text_likeness_passed":r.get("text_likeness_passed"),"text_likeness_reason":r.get("text_likeness_reason"),
          "role":role,"importance":round(importance,1),
          "position":{
            "x_pct":round(box[0]/max(1,W)*100,1),
            "y_pct":round(box[1]/max(1,H)*100,1),
            "width_pct":round(wr*100,1),
            "height_pct":round(hr*100,1),
          },
          "font_estimate":{"top":top_font,"probabilities":fam},
          "color":col,
        })
    out.sort(key=lambda q:(0 if q["role"]=="PRIMARY" else 1 if q["role"]=="SECONDARY" else 2,-q["importance"]))
    return out

def image_design_profile(raw,regions):
    img=decode_gray(raw)
    if img is None:return {}
    small=cv2.resize(img,(160,160),interpolation=cv2.INTER_AREA)
    edge_density=float(np.mean(cv2.Canny(small,60,150)>0))
    palette=palette_profile(raw)
    complexity=min(10.0,edge_density*50+max(0,len(palette.get("dominant",[]))-2)*0.45)
    return {
      "palette":palette,
      "visual_complexity_10":round(complexity,1),
      "primary_text_count":sum(1 for r in regions if r.get("role")=="PRIMARY"),
      "secondary_text_count":sum(1 for r in regions if r.get("role")=="SECONDARY"),
      "text_region_count":len(regions),
    }

def analyze_registration_image(full_raw,registration_name,display_name,detected_texts=None):
    detected=normalize_detected_texts(detected_texts or [])
    img0=decode_color(full_raw)
    image_h=img0.shape[0] if img0 is not None else 1

    likeness_checked=apply_text_likeness_gate(
        detected,full_raw,registration_name,display_name
    )
    likeness_passed=[x for x in likeness_checked if x.get("text_likeness_passed")]

    gated_detected=apply_text_size_gate(
        likeness_passed,image_h,registration_name,display_name
    )

    # audit rows: keep excluded non-text OCR visible in registration diagnostics
    final_map={}
    for x in likeness_checked:
        bb=x["bbox"]; key=(x["text"],bb["x0"],bb["y0"],bb["x1"],bb["y1"])
        final_map[key]=x
    for x in gated_detected:
        bb=x["bbox"]; key=(x["text"],bb["x0"],bb["y0"],bb["x1"],bb["y1"])
        if key in final_map: final_map[key].update(x)
        else: final_map[key]=x
    audit_rows=list(final_map.values())

    text_identity=region_identity_profiles(full_raw,audit_rows)
    primary_raw,secondary_raw=detect_text_bands(full_raw)

    context_base=context_recognition_terms(registration_name,display_name)
    final_passed=[
        x for x in gated_detected
        if x.get("size_gate_passed") and x.get("text_likeness_passed",True)
    ]
    image_terms=detected_text_terms(
        final_passed,image_h,registration_name,display_name
    )
    recognition_terms=list(dict.fromkeys(context_base+image_terms))

    p={
      "profile_version":"4.0",
      "source":"BROWSER_OCR_PLUS_VISUAL_IDENTITY",
      "context":{
        "registration_name":registration_name,
        "display_name":display_name,
        "hint_tokens":context_tokens(registration_name,display_name),
        "image_ocr_terms":image_terms,
        "recognition_terms":recognition_terms,
        "note":"등록명/표시명은 힌트, image_ocr_terms는 등록 이미지에서 실제 검출한 문자."
      },
      "text_identity":text_identity,
      "regions":{
        "primary":region_profile(primary_raw) if primary_raw else {},
        "secondary":region_profile(secondary_raw) if secondary_raw else {}
      },
      "design":image_design_profile(full_raw,text_identity),
      "full_visual":{
        "color":palette_profile(full_raw),
        "typography":typography_features(full_raw),
        "sift":sift_fingerprint(full_raw),
        "multiview_sift":build_multiview_sift(full_raw),
        "variants":generate_variants(full_raw)
      },
      "mode":"OCR_LAYOUT_VISUAL"
    }
    return p,primary_raw,secondary_raw

def registration_analysis_summary(profile):
    return {
      "mode":profile.get("mode"),
      "hint_tokens":profile.get("context",{}).get("hint_tokens",[]),
      "image_ocr_terms":profile.get("context",{}).get("image_ocr_terms",[]),
      "recognition_terms":profile.get("context",{}).get("recognition_terms",[]),
      "text_identity":profile.get("text_identity",[])[:20],
      "design":profile.get("design",{}),
      "full_color":profile.get("full_visual",{}).get("color",{}),
      "sift_features":profile.get("full_visual",{}).get("sift",{}).get("count",0),
      "ocr_ready":bool(profile.get("text_identity",[])),
      "text_likeness_gate":{
        "min_score":TEXT_LIKENESS_MIN_SCORE,
        "passed":sum(1 for x in profile.get("text_identity",[]) if x.get("text_likeness_passed")),
        "excluded":sum(1 for x in profile.get("text_identity",[]) if x.get("text_likeness_passed") is False),
      },
      "text_size_gate":{
        "hard_pct":TEXT_SIZE_GATE_PCT,
        "soft_min_pct":TEXT_SIZE_SOFT_MIN_PCT,
        "soft_conf":TEXT_SIZE_SOFT_CONF,
        "passed":sum(1 for x in profile.get("text_identity",[]) if x.get("size_gate_passed")),
        "excluded":sum(1 for x in profile.get("text_identity",[]) if x.get("size_gate_passed") is False),
      },
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




def perspective_variant(raw,direction):
    img=decode_color(raw)
    if img is None:return raw
    h,w=img.shape[:2]
    if w<20 or h<20:return raw
    k=max(4,int(min(w,h)*0.12))
    src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    if direction=="up15":
        dst=np.float32([[k,0],[w-1-k,0],[w-1,h-1],[0,h-1]])
    elif direction=="down15":
        dst=np.float32([[0,0],[w-1,0],[w-1-k,h-1],[k,h-1]])
    elif direction=="left15":
        dst=np.float32([[0,k],[w-1,0],[w-1,h-1],[0,h-1-k]])
    elif direction=="right15":
        dst=np.float32([[0,0],[w-1,k],[w-1,h-1-k],[0,h-1]])
    else:
        return raw
    H=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(img,H,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    ok,enc=cv2.imencode(".jpg",warped,[cv2.IMWRITE_JPEG_QUALITY,90])
    return enc.tobytes() if ok else raw

def build_multiview_sift(raw):
    variants={
        "front":raw,
        "up15":perspective_variant(raw,"up15"),
        "down15":perspective_variant(raw,"down15"),
        "left15":perspective_variant(raw,"left15"),
        "right15":perspective_variant(raw,"right15"),
    }
    return {k:sift_fingerprint(v) for k,v in variants.items() if v}

def estimate_view_priority(frame_raw):
    gray=decode_gray(frame_raw)
    if gray is None:return ["front","left15","right15","up15","down15"]
    gray,_=_resize_for_features(gray,420)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    h,w=gray.shape[:2]
    left=float(np.mean(np.abs(gx[:,:max(1,w//3)])))
    right=float(np.mean(np.abs(gx[:,max(0,2*w//3):])))
    top=float(np.mean(np.abs(gy[:max(1,h//3),:])))
    bottom=float(np.mean(np.abs(gy[max(0,2*h//3):,:])))
    h1="left15" if left>right else "right15"
    h2="right15" if h1=="left15" else "left15"
    v1="up15" if top>bottom else "down15"
    v2="down15" if v1=="up15" else "up15"
    return ["front",h1,v1,h2,v2]

def multiview_sift_score(multiview_fp,frame_raw):
    priority=estimate_view_priority(frame_raw)
    best=None
    tested=0
    for view in priority:
        fp=multiview_fp.get(view)
        if not fp:continue
        tested+=1
        d=sift_homography_score_from_fp(fp,frame_raw)
        d["view"]=view
        d["tested_views"]=tested
        if best is None or d.get("score",0)>best.get("score",0):
            best=d
        if d.get("score",0)>=90 and d.get("inliers",0)>=24 and d.get("inlier_ratio",0)>=0.75:
            d["early_stop"]=True
            return d
        if tested>=3 and best and best.get("score",0)>=78:
            break
    if best is None:
        best={"score":0.0,"view":"none","good_matches":0,"inliers":0,"inlier_ratio":0.0,"median_error":None,"coverage":0.0,"homography":False}
    best["tested_views"]=tested
    best["early_stop"]=False
    return best

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


@lru_cache(maxsize=512)
def _load_sift_npz_cached(npz_b64):
    import io
    raw = base64.b64decode(npz_b64)
    with np.load(io.BytesIO(raw)) as data:
        return {
            "descriptors": data["descriptors"].astype(np.float32),
            "points": data["points"].astype(np.float32),
            "shape": tuple(int(v) for v in data["shape"]),
        }

def _load_sift_fingerprint(fp):
    if not fp or not fp.get("npz_b64"):
        return None
    try:
        return _load_sift_npz_cached(fp["npz_b64"])
    except Exception:
        return None


def prepare_live_sift(frame_raw):
    frame=decode_gray(frame_raw)
    if frame is None:
        return None
    frame,_=_resize_for_features(frame,520)
    sift=cv2.SIFT_create(nfeatures=650,contrastThreshold=0.03,edgeThreshold=12,sigma=1.6)
    k2,d2=sift.detectAndCompute(frame,None)
    if d2 is None or not k2:
        return None
    return {
        "points":np.float32([k.pt for k in k2]),
        "descriptors":d2.astype(np.float32),
        "count":len(k2),
    }

def sift_homography_score_prepared(fp,live):
    ref=_load_sift_fingerprint(fp)
    if ref is None or live is None:
        return {"score":0.0,"good_matches":0,"inliers":0,"inlier_ratio":0.0,
                "median_error":None,"coverage":0.0,"homography":False}

    d1=ref["descriptors"]
    pts1=ref["points"]
    d2=live["descriptors"]
    pts2=live["points"]

    if d2 is None or len(d1)<6 or len(d2)<6:
        return {"score":0.0,"good_matches":0,"inliers":0,"inlier_ratio":0.0,
                "median_error":None,"coverage":0.0,"homography":False}

    matcher=cv2.BFMatcher(cv2.NORM_L2)
    pairs=matcher.knnMatch(d1,d2,k=2)
    good=[]
    for pair in pairs:
        if len(pair)!=2: continue
        mm,nn=pair
        if mm.distance<0.74*nn.distance:
            good.append(mm)

    if len(good)<4:
        weak=min(24.0,len(good)*4.0)
        return {"score":round(weak,1),"good_matches":len(good),"inliers":0,
                "inlier_ratio":0.0,"median_error":None,
                "coverage":round(min(1.0,len(good)/24.0),3),"homography":False}

    src_pts=np.float32([[pts1[m.queryIdx][0],pts1[m.queryIdx][1]] for m in good]).reshape(-1,1,2)
    dst_pts=np.float32([[pts2[m.trainIdx][0],pts2[m.trainIdx][1]] for m in good]).reshape(-1,1,2)

    try:
        H,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,4.0)
    except cv2.error:
        H,mask=None,None

    if H is None or mask is None:
        return {"score":min(32.0,len(good)*2.0),"good_matches":len(good),"inliers":0,
                "inlier_ratio":0.0,"median_error":None,
                "coverage":round(min(1.0,len(good)/35.0),3),"homography":False}

    inlier_mask=mask.ravel().astype(bool)
    inliers=int(inlier_mask.sum())
    inlier_ratio=inliers/max(1,len(good))

    projected=cv2.perspectiveTransform(src_pts,H)
    errors=np.linalg.norm(projected.reshape(-1,2)-dst_pts.reshape(-1,2),axis=1)
    inlier_errors=errors[inlier_mask] if len(errors)==len(inlier_mask) else errors
    median_error=float(np.median(inlier_errors)) if len(inlier_errors) else 99.0

    rh,rw=ref["shape"]
    inlier_ref=src_pts.reshape(-1,2)[inlier_mask]
    coverage=0.0
    if len(inlier_ref)>=3 and rw>0 and rh>0:
        xspan=(float(inlier_ref[:,0].max())-float(inlier_ref[:,0].min()))/rw
        yspan=(float(inlier_ref[:,1].max())-float(inlier_ref[:,1].min()))/rh
        coverage=max(0.0,min(1.0,(xspan*yspan)**0.5))

    match_strength=min(100.0,100.0*inliers/45.0)
    ratio_score=min(100.0,inlier_ratio*100.0)
    error_score=max(0.0,100.0*(1.0-median_error/8.0))
    coverage_score=min(100.0,coverage*145.0)
    score=match_strength*0.34+ratio_score*0.32+error_score*0.20+coverage_score*0.14

    if inliers>=18 and inlier_ratio>=0.70 and median_error<=3.0:
        score=max(score,78.0)
    if inliers>=30 and inlier_ratio>=0.80 and median_error<=2.0:
        score=max(score,88.0)

    return {
        "score":round(min(100.0,score),1),
        "good_matches":int(len(good)),
        "inliers":inliers,
        "inlier_ratio":round(inlier_ratio,3),
        "median_error":round(median_error,2),
        "coverage":round(coverage,3),
        "homography":True,
    }


def sift_homography_score_from_fp(fp, frame_raw):
    ref = _load_sift_fingerprint(fp)
    frame = decode_gray(frame_raw)
    if ref is None or frame is None:
        return {
            "score": 0.0, "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0,
            "median_error": None, "coverage": 0.0, "homography": False
        }

    frame, _ = _resize_for_features(frame, 520)
    sift = cv2.SIFT_create(nfeatures=650, contrastThreshold=0.03, edgeThreshold=12, sigma=1.6)
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


def candidate_score(ocr_texts,row):
    f=row.fields()
    try:p=json.loads(row.identity_profile or "{}")
    except Exception:p={}
    terms=p.get("context",{}).get("recognition_terms") or context_recognition_terms(f["registration_name"],f["display_name"])
    if not terms:terms=[f["registration_name"],f["display_name"]]
    ranked=[]
    for target in terms:
        best=max((best_term_score(t,target),t) for t in ocr_texts) if ocr_texts else (0.0,"")
        ranked.append({"target":target,"score":round(best[0],1),"ocr":best[1]})
    ranked.sort(key=lambda x:x["score"],reverse=True)
    best=ranked[0]["score"] if ranked else 0.0
    second=ranked[1]["score"] if len(ranked)>1 else 0.0
    return {
      "major_score":round(best,1),"major_ocr":ranked[0]["ocr"] if ranked else "",
      "minor_score":round(second,1),"minor_coverage":0.0,"minor_hits":ranked[:5],
      "text_score":round(min(100,best*.82+second*.18),1),
      "eligible":bool(best>=48.0),"context_hits":ranked[:5]
    }


# -------------------------- routes --------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUPER Registry V0.8.0.9</title>
<script src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js"></script>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#f5f6f8;margin:0;color:#17191c}.w{max-width:1180px;margin:22px auto;padding:0 15px}.c{background:#fff;border:1px solid #ddd;border-radius:14px;padding:18px;margin:13px 0}.g{display:grid;grid-template-columns:1fr 1fr;gap:12px}.full{grid-column:1/-1}label{display:block;font-size:13px;color:#555;margin-bottom:5px}input{width:100%;box-sizing:border-box;padding:10px;border:1px solid #ccd1d7;border-radius:8px}button{border:0;border-radius:8px;background:#111;color:#fff;font-weight:700;padding:9px 12px;cursor:pointer}.danger{background:#a51d27}.ghost{background:#555}.muted{font-size:13px;color:#666}.ok{color:#18723a;font-weight:700}.warn{color:#a22;font-weight:700}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}.profile{white-space:pre-wrap;font:12px/1.55 ui-monospace,monospace;background:#101114;color:#eee;padding:11px;border-radius:8px;max-height:420px;overflow:auto}.preview{display:flex;gap:12px;flex-wrap:wrap}.preview img{max-width:330px;max-height:210px;border:1px solid #ddd;border-radius:8px}.identity{display:grid;grid-template-columns:135px 100px 1fr 180px;gap:5px;font-size:12px;border-bottom:1px solid #eee;padding:7px 0}.modal{display:none;position:fixed;inset:0;background:#0008;z-index:30;padding:30px;overflow:auto}.modalbox{max-width:1050px;margin:auto;background:#fff;border-radius:14px;padding:18px}@media(max-width:820px){.g{grid-template-columns:1fr}.full{grid-column:auto}.identity{grid-template-columns:1fr}table{display:block;overflow:auto}}
</style></head><body><div class="w">
<h2>LUPER Registry V0.8.0.9</h2>
<p><b>Context Hint + 실제 이미지 OCR + Layout/Color + SIFT</b></p>
<p class="muted">등록명·표시명은 힌트입니다. 등록 이미지에서 한글+영문을 실제로 읽어 Winning Habit 같은 문구도 후보어로 저장합니다.</p>
<div class="c"><label>관리자 키</label><div style="display:flex;gap:8px"><input id="admin" type="password"><button id="saveKey">키 저장</button></div></div>
<div class="c"><form id="f"><div class="g">
<div><label>등록명 · 분석 힌트</label><input id="regname" required placeholder="이기는 습관 샘앤파커스"></div>
<div><label>표시명 · 링크카드 표시</label><input id="display" required placeholder="이기는 습관"></div>
<div><label>그룹명 · 결과 묶음용</label><input id="groupname" placeholder="예: 살만온족발 / Galaxy / 이기는 습관"></div>
<div><label>액션명 · 카드 기능</label><input id="actionname" placeholder="예: 브랜드 / 메뉴 / 배달주문 / 설명서"></div>
<div class="full"><label>URL</label><input id="url" required placeholder="https://..."></div>
<div class="full"><label>기준 이미지</label><input id="fullImg" type="file" accept="image/*" required></div>
<div class="full"><div id="ocrState" class="muted">사전분석 시 한글+영문 OCR을 실행합니다.</div></div>
<div class="full" style="display:flex;gap:8px"><button type="button" id="analyze">이미지 사전분석</button><button type="submit">분석 결과로 등록</button></div>
</div></form></div>
<div class="c"><h3>이미지 사전분석</h3><div id="analysis" class="muted">아직 분석하지 않았습니다.</div><div id="previews" class="preview"></div></div>
<div class="c"><h3>등록목록</h3><div id="state"></div><div id="list"></div></div>
<div class="c"><h3>최근 LIVE TRACE</h3><button onclick="loadTraces()">최근 50건</button><div id="traces" class="profile">TRACE 대기</div></div>
</div>
<div id="modal" class="modal"><div class="modalbox"><div style="display:flex;justify-content:space-between"><h3 id="modalTitle">분석보기</h3><button class="ghost" onclick="closeModal()">닫기</button></div><div id="modalImage" class="preview"></div><div id="modalBody"></div></div></div>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
$('admin').value=sessionStorage.getItem('luper_admin')||'';$('saveKey').onclick=()=>{sessionStorage.setItem('luper_admin',$('admin').value);refresh()};const ah=()=>({'X-Admin-Key':$('admin').value});
function file64(f){return new Promise((ok,no)=>{if(!f)return ok(null);let r=new FileReader();r.onload=()=>ok(r.result);r.onerror=no;r.readAsDataURL(f)})}
let ocrCache={fileKey:'',items:[]},ocrWorker=null;function fileKey(f){return f?`${f.name}:${f.size}:${f.lastModified}`:''}
async function ensureOcrWorker(){if(ocrWorker)return ocrWorker;$('ocrState').textContent='OCR 엔진 준비 중… 최초 1회 언어데이터 로딩';ocrWorker=await Tesseract.createWorker(['kor','eng'],1,{logger:m=>{if(m.status)$('ocrState').textContent=`OCR ${m.status} ${Math.round((m.progress||0)*100)}%`}});return ocrWorker}
async function browserOcr(){const f=$('fullImg').files[0];if(!f)return[];const key=fileKey(f);if(ocrCache.fileKey===key&&ocrCache.items.length)return ocrCache.items;const w=await ensureOcrWorker();$('ocrState').textContent='한글+영문 실제 문자를 읽는 중…';const r=await w.recognize(f);const lines=(r.data.lines&&r.data.lines.length?r.data.lines:r.data.words)||[];const items=lines.filter(x=>x.text&&String(x.text).trim().length>1&&x.bbox).map(x=>({text:String(x.text).trim(),confidence:Number(x.confidence||0),bbox:{x0:x.bbox.x0,y0:x.bbox.y0,x1:x.bbox.x1,y1:x.bbox.y1}}));ocrCache={fileKey:key,items};$('ocrState').textContent=`OCR 완료 · ${items.length}개 문자영역`;return items}
async function payload(){return{registration_name:$('regname').value.trim(),display_name:$('display').value.trim(),group_name:$('groupname').value.trim(),action_name:$('actionname').value.trim(),url:$('url').value.trim(),reference_image_base64:await file64($('fullImg').files[0]),detected_texts:await browserOcr()}}
function paletteText(p){return(p?.dominant||[]).map(x=>`${x.family} ${x.percent}%`).join(' / ')}
function identityHtml(rows){if(!rows?.length)return'<p class="warn">이미지 OCR Identity 없음 · 재등록 권장</p>';return rows.slice(0,20).map(r=>`<div class="identity"><b>${esc(r.text)}</b><span>${esc(r.script)} · ${esc(r.role)}<br>${r.text_likeness_passed?'✅ 문자성':'❌ 비문자'} ${esc(r.text_likeness_score)}점<br>${esc(r.text_likeness_reason||'')}<br>${r.size_gate_passed?'✅ 크기':'❌ 크기제외'} · 높이 ${esc(r.height_pct)}%<br>${esc(r.size_gate_reason||'')}</span><span>x ${r.position.x_pct}% / y ${r.position.y_pct}% / 가로 ${r.position.width_pct}% / 세로 ${r.position.height_pct}%<br>글자형태 ${esc(JSON.stringify(r.font_estimate?.top))}</span><span>${esc(paletteText(r.color))}</span></div>`).join('')}
$('analyze').onclick=async()=>{if(!$('admin').value)return alert('관리자 키를 저장하세요.');const b=await payload();$('analysis').textContent='Visual Identity 분석 중…';const r=await fetch('/api/analyze_registration',{method:'POST',headers:{'Content-Type':'application/json','X-Admin-Key':$('admin').value},body:JSON.stringify(b)});const x=await r.json();if(!r.ok)return alert(JSON.stringify(x));const sm=x.summary;$('analysis').innerHTML=`<b>등록 힌트</b>: ${esc((sm.hint_tokens||[]).join(' / '))}<br><b>이미지 실제 OCR</b>: ${esc((sm.image_ocr_terms||[]).join(' | '))}<br><b>LIVE 후보어</b>: ${esc((sm.recognition_terms||[]).join(' | '))}<br><b>전체 색상</b>: ${esc(paletteText(sm.full_color))}<br><b>디자인 복잡도</b>: ${esc(sm.design?.visual_complexity_10)}/10 · <b>SIFT</b>: ${sm.sift_features}<br><b>TEXT LIKENESS</b>: 기준 ${esc(sm.text_likeness_gate?.min_score)}점 / 문자 ${esc(sm.text_likeness_gate?.passed)} / 비문자 제외 ${esc(sm.text_likeness_gate?.excluded)}<br><b>TEXT SIZE GATE</b>: ${esc(sm.text_size_gate?.hard_pct)}% 이상 기본 통과 / 통과 ${esc(sm.text_size_gate?.passed)} / 제외 ${esc(sm.text_size_gate?.excluded)}<br><br><b>TEXT IDENTITY</b>${identityHtml(sm.text_identity)}`;$('previews').innerHTML=(x.primary_preview?`<div>PRIMARY 시각밴드<br><img src="${x.primary_preview}"></div>`:'')+(x.secondary_preview?`<div>SECONDARY 시각밴드<br><img src="${x.secondary_preview}"></div>`:'')};
$('f').onsubmit=async e=>{e.preventDefault();if(!$('admin').value)return alert('관리자 키를 저장하세요.');const b=await payload();const r=await fetch('/api/entries',{method:'POST',headers:{'Content-Type':'application/json','X-Admin-Key':$('admin').value},body:JSON.stringify(b)});const x=await r.json();if(!r.ok)return alert(JSON.stringify(x));alert('등록 완료');refresh()};
async function refresh(){const h=await(await fetch('/health')).json();$('state').innerHTML=`<span class="ok">ONLINE</span> V${h.version} · ${h.entries}개`;const r=await fetch('/api/entries',{headers:ah()});if(!r.ok)return;const a=await r.json();let t='<table><tr><th>등록명</th><th>표시명</th><th>그룹/액션</th><th>이미지 OCR</th><th>색/Visual</th><th>URL</th><th>관리</th></tr>';for(const x of a){const o=x.profile_summary?.image_ocr_terms||[];t+=`<tr><td><b>${esc(x.registration_name)}</b></td><td>${esc(x.display_name)}</td><td>${esc(x.group_name||'')} / ${esc(x.action_name||'')}</td><td>${x.profile_summary?.ocr_ready?esc(o.slice(0,7).join(' / ')):'<span class="warn">OCR 미분석 · 재등록 권장</span>'}</td><td>${esc(paletteText(x.profile_summary?.color))}<br>SIFT ${esc(x.profile_summary?.sift_features||0)}</td><td><a href="${esc(x.url)}" target="_blank">열기</a></td><td><button onclick="showProfile(${x.id})">분석보기</button> <button class="danger" onclick="delx(${x.id})">삭제</button></td></tr>`}$('list').innerHTML=t+'</table>'}
async function showProfile(id){const r=await fetch('/api/entries/'+id+'/profile',{headers:ah()});if(!r.ok)return alert('분석정보 조회 실패');const x=await r.json(),p=x.profile||{};$('modalTitle').textContent=`${x.registration_name} · 분석정보`;const ir=await fetch('/api/entries/'+id+'/reference-image',{headers:ah()});if(ir.ok){const u=URL.createObjectURL(await ir.blob());$('modalImage').innerHTML=`<div>등록 기준이미지<br><img src="${u}"></div>`}else $('modalImage').innerHTML='';$('modalBody').innerHTML=`<p><b>그룹/액션:</b> ${esc(x.group_name||'')} / ${esc(x.action_name||'')}</p><p><b>등록 힌트:</b> ${esc((p.context?.hint_tokens||[]).join(' / '))}</p><p><b>이미지 OCR:</b> ${esc((p.context?.image_ocr_terms||[]).join(' | '))}</p><p><b>전체 후보어:</b> ${esc((p.context?.recognition_terms||[]).join(' | '))}</p><p><b>색상:</b> ${esc(paletteText(p.full_visual?.color))} · <b>디자인:</b> ${esc(p.design?.visual_complexity_10)}/10</p>${identityHtml(p.text_identity||[])}<h4>Raw Identity Profile</h4><div class="profile">${esc(JSON.stringify(p,null,2))}</div>`;$('modal').style.display='block'}
function closeModal(){$('modal').style.display='none';$('modalImage').innerHTML=''}
async function delx(id){if(!confirm('이 등록을 삭제할까요?'))return;const r=await fetch('/api/entries/'+id,{method:'DELETE',headers:ah()});if(!r.ok)return alert('삭제 실패');refresh()}
async function loadTraces(){const r=await fetch('/api/traces?limit=50',{headers:ah()});if(!r.ok)return;$('traces').textContent=(await r.json()).map(x=>`${x.created_at||''} | ${x.session_id} | ${x.event}\n${JSON.stringify(x.payload)}`).join('\n\n')}
$('fullImg').addEventListener('change',()=>{ocrCache={fileKey:'',items:[]};$('ocrState').textContent='새 이미지 선택됨 · 사전분석 시 한글+영문 OCR 실행'});refresh();
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


@app.post("/api/analyze_registration")
def analyze_registration():
    if not require_admin():return jsonify(error="unauthorized"),401
    x=request.get_json(silent=True) or {}
    rn=str(x.get("registration_name","")).strip();dn=str(x.get("display_name","")).strip()
    raw=b64_bytes(x.get("reference_image_base64"))
    if not rn or not dn or not raw or decode_gray(raw) is None:
        return jsonify(error="등록명/표시명/기준이미지가 필요합니다."),400
    p,pr,sr=analyze_registration_image(raw,rn,dn,x.get("detected_texts") or [])
    enc=lambda z:("data:image/jpeg;base64,"+base64.b64encode(z).decode("ascii")) if z else None
    return jsonify(ok=True,version=VERSION,summary=registration_analysis_summary(p),primary_preview=enc(pr),secondary_preview=enc(sr))

@app.post("/api/entries")
def create_entry():
    if not require_admin():return jsonify(error="unauthorized"),401
    x=request.get_json(silent=True) or {}
    rn=str(x.get("registration_name","")).strip();dn=str(x.get("display_name","")).strip();url=str(x.get("url","")).strip()
    group_name=str(x.get("group_name","")).strip()
    action_name=str(x.get("action_name","")).strip()
    raw=b64_bytes(x.get("reference_image_base64"))
    if not rn or not dn or not url or not raw:return jsonify(error="등록명/표시명/URL/기준이미지는 필수입니다."),400
    if decode_gray(raw) is None:return jsonify(error="기준이미지를 읽을 수 없습니다."),400
    p,pr,sr=analyze_registration_image(raw,rn,dn,x.get("detected_texts") or [])
    hints=p.get("context",{}).get("recognition_terms",[])
    item=Link(key_text=rn,registration_name=rn,major_category="",minor_category="",recognition_text=" | ".join(hints),
      display_name=dn,group_name=(group_name or rn),action_name=action_name,url=url,match_mode="CONTEXT_VISUAL",reference_image=raw,major_reference=pr,minor_reference=sr,
      visual_threshold=48.0,priority=int(x.get("priority") or 0),use_location=bool(x.get("use_location")),
      latitude=float(x["latitude"]) if x.get("latitude") is not None else None,
      longitude=float(x["longitude"]) if x.get("longitude") is not None else None,
      radius_m=float(x.get("radius_m") or 150),identity_profile=json.dumps(p,ensure_ascii=False))
    db.session.add(item);db.session.commit();return jsonify(item.public_dict()),201


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



@app.get("/api/entries/<int:item_id>/profile")
def entry_profile(item_id):
    if not require_admin(): return jsonify(error="unauthorized"),401
    item=db.session.get(Link,item_id)
    if not item:return jsonify(error="not found"),404
    try:p=json.loads(item.identity_profile or "{}")
    except Exception:p={}
    f=item.fields()
    return jsonify(id=item.id,registration_name=f["registration_name"],display_name=f["display_name"],group_name=(item.group_name or f["registration_name"]),action_name=(item.action_name or ""),url=item.url,profile=p)

@app.get("/api/entries/<int:item_id>/reference-image")
def entry_reference_image(item_id):
    if not require_admin(): return jsonify(error="unauthorized"),401
    item=db.session.get(Link,item_id)
    if not item or not item.reference_image:return jsonify(error="not found"),404
    return Response(item.reference_image,mimetype="image/jpeg")

@app.get("/api/index")
def registry_index():
    rows=Link.query.filter_by(enabled=True).order_by(Link.priority.desc(),Link.id.desc()).all()
    data=[]
    for r in rows:
        f=r.fields()
        try:p=json.loads(r.identity_profile or "{}")
        except Exception:p={}
        terms=p.get("context",{}).get("recognition_terms") or context_recognition_terms(f["registration_name"],f["display_name"])
        data.append({"id":r.id,"registration_name":f["registration_name"],"display_name":f["display_name"],"group_name":(r.group_name or f["registration_name"]),"action_name":(r.action_name or ""),"recognition_terms":terms,"priority":r.priority,"match_mode":r.match_mode or "CONTEXT_VISUAL"})
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




@app.post("/api/qr_fast_verify")
def qr_fast_verify():
    started=time.perf_counter()
    x=request.get_json(silent=True) or {}
    candidate_ids=[int(v) for v in (x.get("candidate_ids") or [])[:6]]
    frame_raw=b64_bytes(x.get("image_base64"))
    visual_only=bool(x.get("visual_only"))
    lat,lon=x.get("lat"),x.get("lon")

    if not candidate_ids or not frame_raw:
        return jsonify(matches=[],diagnostics=[],elapsed_ms=0,version=VERSION)

    matches=[]
    diagnostics=[]
    live_sift=prepare_live_sift(frame_raw)

    for cid in candidate_ids:
        row=db.session.get(Link,cid)
        if not row or not row.enabled:
            continue

        try:
            p=json.loads(row.identity_profile or "{}")
        except Exception:
            p={}

        fp=p.get("full_visual",{}).get("sift",{})
        if not fp:
            fp=sift_fingerprint(row.reference_image)

        sd=sift_homography_score_prepared(fp,live_sift)
        visual=float(sd.get("score",0.0))
        gok,dist=gps_ok(row,lat,lon)

        median_error=sd.get("median_error")
        coverage=float(sd.get("coverage",0) or 0)
        good_matches=int(sd.get("good_matches",0) or 0)
        inliers=int(sd.get("inliers",0) or 0)
        ratio=float(sd.get("inlier_ratio",0) or 0)

        # Text-supported candidate can use a normal strong visual threshold.
        text_supported=bool(
            sd.get("homography")
            and visual>=88.0
            and good_matches>=22
            and inliers>=18
            and ratio>=0.70
            and coverage>=0.12
            and (median_error is None or median_error<=3.0)
        )

        # No OCR/context evidence: demand near-identity geometry.
        # This prevents a pizza from becoming an 83% "Winning Habit" false positive.
        visual_only_supported=bool(
            sd.get("homography")
            and visual>=94.0
            and good_matches>=35
            and inliers>=28
            and ratio>=0.78
            and coverage>=0.22
            and (median_error is None or median_error<=2.0)
        )

        strong=visual_only_supported if visual_only else text_supported

        f=row.fields()
        d={
            "id":row.id,
            "registration_name":f["registration_name"],
            "display_name":f["display_name"],
            "group_name":(row.group_name or f["registration_name"]),
            "action_name":(row.action_name or ""),
            "url":row.url,
            "visual_method":"FRONT_SIFT_QR_FAST",
            "matched_view":"front",
            "tested_views":1,
            "burst_frame":x.get("selected_frame",1),
            "burst_frames_tested":1,
            "visual_score":round(visual,1),
            "sift_good_matches":sd.get("good_matches",0),
            "sift_inliers":sd.get("inliers",0),
            "sift_inlier_ratio":sd.get("inlier_ratio",0),
            "sift_median_error":sd.get("median_error"),
            "sift_coverage":sd.get("coverage",0),
            "gps_ok":gok,
            "distance_m":dist,
            "passed":bool(strong and gok),
            "final_confidence":round(visual,1),
            "fast_path":True,
            "visual_only":visual_only,
            "text_supported":text_supported,
            "visual_only_supported":visual_only_supported,
            "required_visual_score":94.0 if visual_only else 88.0,
        }
        diagnostics.append(d)
        if d["passed"]:
            matches.append(d)

    matches.sort(key=lambda q:q["visual_score"],reverse=True)
    elapsed=int((time.perf_counter()-started)*1000)
    return jsonify(matches=matches,diagnostics=diagnostics,elapsed_ms=elapsed,version=VERSION)


@app.post("/api/fast_verify")
def fast_verify():
    started = datetime.now(timezone.utc)
    x = request.get_json(silent=True) or {}
    candidate_ids = [int(v) for v in (x.get("candidate_ids") or [])[:3]]

    frame_items = x.get("frames") or []
    frame_raws = []
    for item in frame_items[:3]:
        if isinstance(item, dict):
            raw = b64_bytes(item.get("image_base64"))
        else:
            raw = b64_bytes(item)
        if raw:
            frame_raws.append(raw)

    # backward compatibility
    if not frame_raws:
        single = b64_bytes(x.get("image_base64"))
        if single:
            frame_raws=[single]

    lat, lon = x.get("lat"), x.get("lon")
    if not candidate_ids or not frame_raws:
        return jsonify(matches=[], diagnostics=[], elapsed_ms=0, version=VERSION)

    matches=[]
    diagnostics=[]

    for cid in candidate_ids:
        row=db.session.get(Link,cid)
        if not row or not row.enabled:
            continue

        try:
            p=json.loads(row.identity_profile or "{}")
        except Exception:
            p={}
        mv=p.get("full_visual",{}).get("multiview_sift",{})
        if not mv:
            mv=build_multiview_sift(row.reference_image)

        best=None
        frames_tested=0
        for fi,frame_raw in enumerate(frame_raws):
            frames_tested += 1
            sd=multiview_sift_score(mv,frame_raw)
            sd["burst_frame"]=fi+1
            if best is None or sd.get("score",0)>best.get("score",0):
                best=sd

            # First-exposure fast stop:
            # one frame with very strong geometry is enough.
            if (
                sd.get("score",0)>=90.0
                and sd.get("inliers",0)>=24
                and sd.get("inlier_ratio",0)>=0.75
            ):
                sd["burst_early_stop"]=True
                best=sd
                break

        sd=best or {}
        visual=float(sd.get("score",0.0))
        gok,dist=gps_ok(row,lat,lon)

        strong=bool(
            sd.get("homography")
            and sd.get("inliers",0)>=18
            and sd.get("inlier_ratio",0)>=0.68
            and visual>=80.0
        )

        f=row.fields()
        d={
            "id":row.id,
            "registration_name":f["registration_name"],
            "display_name":f["display_name"],
            "major_category":f["major_category"],
            "minor_category":f["minor_category"],
            "url":row.url,
            "visual_method":"MULTIVIEW_BURST_SIFT",
            "matched_view":sd.get("view","front"),
            "tested_views":sd.get("tested_views",0),
            "early_stop":sd.get("early_stop",False),
            "burst_frame":sd.get("burst_frame",1),
            "burst_frames_tested":frames_tested,
            "burst_early_stop":sd.get("burst_early_stop",False),
            "visual_score":round(visual,1),
            "sift_good_matches":sd.get("good_matches",0),
            "sift_inliers":sd.get("inliers",0),
            "sift_inlier_ratio":sd.get("inlier_ratio",0),
            "sift_median_error":sd.get("median_error"),
            "sift_coverage":sd.get("coverage",0),
            "gps_ok":gok,
            "distance_m":dist,
            "passed":bool(strong and gok),
            "final_confidence":round(visual,1),
            "fast_path":True,
        }
        diagnostics.append(d)
        if d["passed"]:
            matches.append(d)

    matches.sort(key=lambda q:q["visual_score"],reverse=True)
    elapsed=int((datetime.now(timezone.utc)-started).total_seconds()*1000)
    return jsonify(matches=matches,diagnostics=diagnostics,elapsed_ms=elapsed,version=VERSION)


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
        ("group_name", "VARCHAR(300)"),
        ("action_name", "VARCHAR(300)"),
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
          match_mode=COALESCE(NULLIF(match_mode,''),'auto'),
          group_name=COALESCE(NULLIF(group_name,''),registration_name,display_name,key_text),
          action_name=COALESCE(action_name,'')
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
            not current or current.get("profile_version") not in ("3.0","4.0") or
            not current.get("full_visual", {}).get("sift", {}).get("npz_b64") or
            not current.get("full_visual", {}).get("multiview_sift")
        )
        if needs_rebuild:
            try:
                p,pr,sr=analyze_registration_image(row.reference_image,f["registration_name"],f["display_name"])
                row.identity_profile=json.dumps(p,ensure_ascii=False)
                row.major_reference=pr;row.minor_reference=sr
                row.match_mode="CONTEXT_VISUAL"
                row.recognition_text=" | ".join(p.get("context",{}).get("recognition_terms",[]))
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
