import base64, difflib, math, os, re
from datetime import datetime, timezone
import cv2, numpy as np
from flask import Flask, jsonify, request, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text as sql_text
app=Flask(__name__)
db_url=os.getenv("DATABASE_URL","sqlite:///registry.db")
if db_url.startswith("postgres://"): db_url="postgresql+psycopg://"+db_url[len("postgres://"):]
elif db_url.startswith("postgresql://") and "+psycopg" not in db_url: db_url="postgresql+psycopg://"+db_url[len("postgresql://"):]
app.config["SQLALCHEMY_DATABASE_URI"]=db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]=False
app.config["MAX_CONTENT_LENGTH"]=10*1024*1024
db=SQLAlchemy(app);ADMIN_KEY=os.getenv("ADMIN_KEY","change-me");VERSION="0.4.0"
class Link(db.Model):
 __tablename__="links"
 id=db.Column(db.Integer,primary_key=True);key_text=db.Column(db.String(300),nullable=False,default="");registration_name=db.Column(db.String(300));major_category=db.Column(db.String(300));minor_category=db.Column(db.String(500));recognition_text=db.Column(db.String(1000));display_name=db.Column(db.String(300),nullable=False);url=db.Column(db.Text,nullable=False);use_location=db.Column(db.Boolean,nullable=False,default=False);latitude=db.Column(db.Float);longitude=db.Column(db.Float);radius_m=db.Column(db.Float,nullable=False,default=150.0);priority=db.Column(db.Integer,nullable=False,default=0);enabled=db.Column(db.Boolean,nullable=False,default=True);visual_threshold=db.Column(db.Float,nullable=False,default=55.0);reference_image=db.Column(db.LargeBinary,nullable=False);created_at=db.Column(db.DateTime(timezone=True),nullable=False,default=lambda:datetime.now(timezone.utc))
 def normalized_fields(self):
  major=(self.major_category or self.key_text or "").strip();return {"registration_name":(self.registration_name or self.display_name or major).strip(),"major_category":major,"minor_category":(self.minor_category or "").strip(),"recognition_text":(self.recognition_text or self.key_text or major).strip(),"display_name":(self.display_name or major).strip()}
 def public_dict(self):
  f=self.normalized_fields();return {"id":self.id,**f,"key_text":self.key_text,"url":self.url,"use_location":bool(self.use_location),"latitude":self.latitude,"longitude":self.longitude,"radius_m":self.radius_m,"priority":self.priority,"enabled":bool(self.enabled),"visual_threshold":self.visual_threshold,"reference_image":True,"created_at":self.created_at.isoformat() if self.created_at else None}
def norm(s): return "".join(ch.lower() for ch in (s or "") if ch.isalnum())
def sim(a,b):
 a,b=norm(a),norm(b)
 if not a or not b:return 0.0
 if a==b:return 100.0
 if a in b or b in a:return 88.0+12.0*min(len(a),len(b))/max(len(a),len(b))
 return 100.0*difflib.SequenceMatcher(None,a,b).ratio()
def tokens(s): return [t for t in re.split(r"[\s,|/·]+",s or "") if norm(t)]
def token_best(ocr,field):
 ts=tokens(field);return max([sim(ocr,t) for t in ts],default=0.0)
def text_score(ocr,row):
 f=row.normalized_fields();major=sim(ocr,f["major_category"]);minor=max(sim(ocr,f["minor_category"]),token_best(ocr,f["minor_category"]));recog=max(sim(ocr,f["recognition_text"]),token_best(ocr,f["recognition_text"]));labels=max(sim(ocr,f["display_name"]),sim(ocr,f["registration_name"]));weighted=major*.45+minor*.25+recog*.20+labels*.10
 if norm(f["major_category"]) and (norm(f["major_category"]) in norm(ocr) or norm(ocr) in norm(f["major_category"])): weighted=max(weighted,82.0)
 return round(min(100.0,weighted),1),{"major":round(major,1),"minor":round(minor,1),"recognition":round(recog,1),"labels":round(labels,1)}
def hav(a,b,c,d):
 r=6371000.;p1,p2=math.radians(a),math.radians(c);dp=math.radians(c-a);dl=math.radians(d-b);x=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2;return 2*r*math.asin(math.sqrt(x))
def b64_bytes(data):
 if not data:return None
 raw=data.split(",",1)[1] if "," in data else data
 try:return base64.b64decode(raw,validate=False)
 except:return None
def decode_gray(raw): return cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE) if raw else None
def prep(gray):
 if gray is None:return None
 blur=cv2.GaussianBlur(gray,(3,3),0);_,bw=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU);pts=cv2.findNonZero(bw)
 if pts is not None:
  x,y,w,h=cv2.boundingRect(pts);px=max(4,int(w*.08));py=max(4,int(h*.12));bw=bw[max(0,y-py):min(gray.shape[0],y+h+py),max(0,x-px):min(gray.shape[1],x+w+px)]
 return bw if bw is not None and bw.size else None
def shape_score(ref,frame):
 a,b=prep(ref),prep(frame)
 if a is None or b is None:return 0.0
 W,H=640,240
 def fit(src):
  h,w=src.shape[:2];scale=min(W/max(1,w),H/max(1,h));nw,nh=max(1,int(w*scale)),max(1,int(h*scale));r=cv2.resize(src,(nw,nh),interpolation=cv2.INTER_AREA);c=np.zeros((H,W),np.uint8);x=(W-nw)//2;y=(H-nh)//2;c[y:y+nh,x:x+nw]=r;return c
 aa,bb=fit(a),fit(b);corr=max(0.0,float(cv2.matchTemplate(aa,bb,cv2.TM_CCOEFF_NORMED)[0][0]));ea,eb=cv2.Canny(aa,50,140),cv2.Canny(bb,50,140);inter=np.logical_and(ea>0,eb>0).sum();union=np.logical_or(ea>0,eb>0).sum();iou=float(inter/union) if union else 0.0;return min(100.,100*(.78*corr+.22*iou))
def orb_score(ref,frame):
 orb=cv2.ORB_create(nfeatures=1800,fastThreshold=6);k1,d1=orb.detectAndCompute(ref,None);k2,d2=orb.detectAndCompute(frame,None)
 if d1 is None or d2 is None or len(k1)<6 or len(k2)<6:return 0.0
 pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2);good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.74*n.distance]
 if len(good)<4:return min(30.,len(good)*6.)
 coverage=min(1.,len(good)/max(14.,min(len(k1),45.)));inlier=0.
 if len(good)>=6:
  src=np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2);dst=np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
  try:
   _,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0);inlier=float(mask.sum())/len(mask) if mask is not None and len(mask) else 0.
  except cv2.error:pass
 return min(100.,100*(.56*coverage+.44*inlier))
def visual_components(rb,fb):
 ref,frame=decode_gray(rb),decode_gray(fb)
 if ref is None or frame is None:return 0.,0.,0.
 sh=shape_score(ref,frame);orb=orb_score(ref,frame);final=max(sh,.58*sh+.42*orb);return round(sh,1),round(orb,1),round(min(100.,final),1)
def require_admin(): return bool(ADMIN_KEY) and request.headers.get("X-Admin-Key","")==ADMIN_KEY
INDEX_HTML='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LUPER Registry V0.4.1</title><style>\nbody{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f5f6f8;color:#17191c}.w{max-width:1180px;margin:24px auto;padding:0 16px}.c{background:#fff;border:1px solid #ddd;border-radius:14px;padding:18px;margin:14px 0}.g{display:grid;grid-template-columns:1fr 1fr;gap:12px}.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}label{display:block;font-size:13px;color:#555;margin:0 0 5px}input,select{width:100%;box-sizing:border-box;padding:10px;border:1px solid #cfd4da;border-radius:8px;font-size:14px}.full{grid-column:1/-1}.row{display:flex;gap:8px;align-items:center}.row input[type=checkbox]{width:auto}button{padding:10px 13px;border:0;border-radius:8px;background:#111;color:#fff;font-weight:700;cursor:pointer}.muted{color:#666;font-size:13px}.warn{color:#a33;font-weight:700}.ok{color:#176b36;font-weight:700}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}a{color:#1558d6}.diag{white-space:pre-wrap;background:#111;color:#eee;padding:12px;border-radius:8px;font:12px/1.5 ui-monospace,monospace}@media(max-width:800px){.g,.g3{grid-template-columns:1fr}.full{grid-column:auto}table{display:block;overflow:auto}}\n</style></head><body><div class="w"><h2>LUPER Visual Link Registry V0.4.1</h2><p><b>TEXT 선별 → 상위 후보만 VISUAL 검증 → GPS → 5초 링크카드</b></p>\n<div class="c"><label>관리자 키</label><div class="row"><input id="admin" type="password" placeholder="Render ADMIN_KEY" style="flex:1"><button type="button" id="saveKey">키 저장</button></div><p class="muted">브라우저 탭의 sessionStorage에만 저장됩니다.</p></div>\n<div class="c"><form id="f"><div class="g"><div><label>등록명 (관리자용 고유 이름)</label><input id="regname" required placeholder="살만온족발_메뉴_01"></div><div><label>표시명 (카드에 표시)</label><input id="display" required placeholder="살만온족발 메뉴"></div><div><label>대분류 · 상호/브랜드</label><input id="major" required placeholder="살만온족발"></div><div><label>소분류 · 서비스/콘텐츠</label><input id="minor" placeholder="족발 보쌈 닭발"></div><div class="full"><label>인식문자 · OCR 보조 키워드</label><input id="recognition" placeholder="살만온족발 족발 보쌈 닭발"></div><div class="full"><label>URL</label><input id="url" required placeholder="https://..."></div><div class="full"><label>실제 글자/간판 기준 이미지 (필수)</label><input id="img" type="file" accept="image/*" required></div><div><label>Visual 최소점수</label><input id="thr" type="number" min="0" max="100" value="55"></div><div><label>우선순위</label><input id="pri" type="number" value="0"></div><div class="full row"><input id="use" type="checkbox"><label for="use" style="margin:0">GPS 조건 사용</label></div><div><label>위도</label><input id="lat" type="number" step="any"></div><div><label>경도</label><input id="lon" type="number" step="any"></div><div><label>반경(m)</label><input id="rad" type="number" value="150"></div><div class="full"><button>등록</button></div></div></form></div>\n<div class="c"><h3>Visual 진단</h3><p class="muted">등록 원본과 실제 촬영 이미지의 Shape / ORB / Final Visual을 분리해서 확인합니다.</p><div class="g3"><div><label>등록항목</label><select id="diagEntry"></select></div><div><label>비교 이미지</label><input id="diagImg" type="file" accept="image/*"></div><div style="align-self:end"><button id="diagBtn" type="button">Visual 진단</button></div></div><div id="diagResult" class="diag" style="margin-top:12px">진단 대기</div></div>\n<div class="c"><div id="state" class="muted">서버 확인 중…</div><div id="list"></div></div><div class="c muted">TEXT 가중치: 대분류 45% · 소분류 25% · 인식문자 20% · 표시/등록명 10%. TEXT 상위 3개 후보만 Visual 비교합니다.</div></div>\n<script>\nconst $=id=>document.getElementById(id);const esc=s=>String(s??\'\').replace(/[&<>"]/g,c=>({\'&\':\'&amp;\',\'<\':\'&lt;\',\'>\':\'&gt;\',\'"\':\'&quot;\'}[c]));$(\'admin\').value=sessionStorage.getItem(\'luper_admin_key\')||\'\';$(\'saveKey\').onclick=()=>{sessionStorage.setItem(\'luper_admin_key\',$(\'admin\').value);refresh();};const ah=()=>({\'X-Admin-Key\':$(\'admin\').value});function file64(f){return new Promise((ok,no)=>{let r=new FileReader();r.onload=()=>ok(r.result);r.onerror=no;r.readAsDataURL(f)})}\nasync function refresh(){try{const hs=await (await fetch(\'/health\')).json();$(\'state\').innerHTML=`<span class="ok">ONLINE</span> · V${esc(hs.version)} · DB ${esc(hs.database)} · entries ${hs.entries}`;const r=await fetch(\'/api/entries\',{headers:ah()});if(r.status===401){$(\'list\').innerHTML=\'<p class="warn">관리자 키를 저장하면 등록목록이 보입니다.</p>\';return}const a=await r.json();$(\'diagEntry\').innerHTML=a.map(x=>`<option value="${x.id}">${esc(x.registration_name)} · ${esc(x.display_name)}</option>`).join(\'\');let h=\'<table><tr><th>등록명</th><th>대분류</th><th>소분류</th><th>인식문자</th><th>표시명</th><th>URL</th><th>Visual</th><th>GPS</th><th></th></tr>\';for(const x of a){h+=`<tr><td><b>${esc(x.registration_name)}</b></td><td>${esc(x.major_category)}</td><td>${esc(x.minor_category)}</td><td>${esc(x.recognition_text)}</td><td>${esc(x.display_name)}</td><td style="max-width:250px;word-break:break-all"><a href="${esc(x.url)}" target="_blank">${esc(x.url)}</a></td><td>${x.visual_threshold}%</td><td>${x.use_location?x.radius_m+\'m\':\'무관\'}</td><td><button onclick="delx(${x.id})">삭제</button></td></tr>`}$(\'list\').innerHTML=h+\'</table>\'}catch(e){$(\'state\').innerHTML=\'<span class="warn">서버 연결 실패</span>\'}}\nasync function delx(id){if(!confirm(\'삭제할까요?\'))return;const r=await fetch(\'/api/entries/\'+id,{method:\'DELETE\',headers:ah()});if(!r.ok){alert(await r.text());return}refresh()}\n$(\'f\').onsubmit=async e=>{e.preventDefault();if(!$(\'admin\').value){alert(\'관리자 키를 먼저 저장하세요.\');return}const image=await file64($(\'img\').files[0]);const o={registration_name:$(\'regname\').value,display_name:$(\'display\').value,major_category:$(\'major\').value,minor_category:$(\'minor\').value,recognition_text:$(\'recognition\').value,url:$(\'url\').value,reference_image_base64:image,visual_threshold:+$(\'thr\').value||55,priority:+$(\'pri\').value||0,use_location:$(\'use\').checked,latitude:$(\'lat\').value?+$(\'lat\').value:null,longitude:$(\'lon\').value?+$(\'lon\').value:null,radius_m:+$(\'rad\').value||150};const r=await fetch(\'/api/entries\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\',\'X-Admin-Key\':$(\'admin\').value},body:JSON.stringify(o)});if(!r.ok){alert(await r.text());return}$(\'f\').reset();$(\'thr\').value=55;$(\'rad\').value=150;$(\'pri\').value=0;refresh()};\n$(\'diagBtn\').onclick=async()=>{if(!$(\'diagImg\').files[0]){alert(\'비교 이미지를 선택하세요.\');return}const image=await file64($(\'diagImg\').files[0]);$(\'diagResult\').textContent=\'진단 중...\';const r=await fetch(\'/api/visual_diagnostic\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\',\'X-Admin-Key\':$(\'admin\').value},body:JSON.stringify({entry_id:+$(\'diagEntry\').value,test_image_base64:image})});const x=await r.json();if(!r.ok){$(\'diagResult\').textContent=JSON.stringify(x,null,2);return}$(\'diagResult\').textContent=`등록명: ${x.registration_name}\nShape: ${x.shape_score}%\nORB: ${x.orb_score}%\nFinal Visual: ${x.visual_score}%\n현재 기준: ${x.visual_threshold}%\n판정: ${x.visual_score>=x.visual_threshold?\'PASS\':\'FAIL\'}\n\n원본↔원본이 90% 미만이면 알고리즘/기준이미지 문제\n스크린샷은 높고 폰 촬영만 낮으면 원근·반사·Crop 영향`};refresh();\n</script></body></html>'
@app.get("/")
def index():return Response(INDEX_HTML,content_type="text/html; charset=utf-8")
@app.get("/health")
def health():return jsonify(ok=True,version=VERSION,database=db.engine.url.get_backend_name(),entries=Link.query.count())
@app.get("/api/entries")
def entries():
 if not require_admin():return jsonify(error="unauthorized"),401
 return jsonify([x.public_dict() for x in Link.query.filter_by(enabled=True).order_by(Link.priority.desc(),Link.id.desc()).all()])
@app.post("/api/entries")
def create_entry():
 if not require_admin():return jsonify(error="unauthorized"),401
 x=request.get_json(silent=True) or {};needed=("registration_name","display_name","major_category","url","reference_image_base64")
 if not all(str(x.get(k,"")).strip() for k in needed):return jsonify(error="등록명/표시명/대분류/URL/기준이미지는 필수입니다."),400
 raw=b64_bytes(x.get("reference_image_base64"));use=bool(x.get("use_location"))
 if not raw or decode_gray(raw) is None:return jsonify(error="기준이미지를 읽을 수 없습니다."),400
 if use and (x.get("latitude") is None or x.get("longitude") is None):return jsonify(error="GPS 사용 시 위도/경도가 필요합니다."),400
 major=str(x["major_category"]).strip();item=Link(key_text=major,registration_name=str(x["registration_name"]).strip(),major_category=major,minor_category=str(x.get("minor_category","")).strip(),recognition_text=str(x.get("recognition_text","")).strip() or major,display_name=str(x["display_name"]).strip(),url=str(x["url"]).strip(),reference_image=raw,visual_threshold=float(x.get("visual_threshold") or 55),priority=int(x.get("priority") or 0),use_location=use,latitude=float(x["latitude"]) if x.get("latitude") is not None else None,longitude=float(x["longitude"]) if x.get("longitude") is not None else None,radius_m=float(x.get("radius_m") or 150));db.session.add(item);db.session.commit();return jsonify(item.public_dict()),201
@app.delete("/api/entries/<int:item_id>")
def delete_entry(item_id):
 if not require_admin():return jsonify(error="unauthorized"),401
 item=db.session.get(Link,item_id)
 if not item:return jsonify(error="not found"),404
 db.session.delete(item);db.session.commit();return jsonify(ok=True)
@app.post("/api/visual_diagnostic")
def visual_diagnostic():
 if not require_admin():return jsonify(error="unauthorized"),401
 x=request.get_json(silent=True) or {};item=db.session.get(Link,int(x.get("entry_id") or 0));test=b64_bytes(x.get("test_image_base64"))
 if not item or not test:return jsonify(error="entry/image required"),400
 sh,orb,final=visual_components(item.reference_image,test);return jsonify(registration_name=item.normalized_fields()["registration_name"],shape_score=sh,orb_score=orb,visual_score=final,visual_threshold=item.visual_threshold)
@app.post("/api/match_blocks")
def match_blocks():
 x=request.get_json(silent=True) or {};blocks=x.get("blocks") or [];lat,lon=x.get("lat"),x.get("lon")
 if not isinstance(blocks,list) or not blocks:return jsonify(matches=[],diagnostics=[],version=VERSION)
 rows=Link.query.filter_by(enabled=True).all();candidate_map={}
 for bi,block in enumerate(blocks[:12]):
  ocr=str((block or {}).get("text","")).strip();crop=b64_bytes((block or {}).get("crop_jpeg_base64"))
  if len(norm(ocr))<2 or not crop:continue
  for r in rows:
   ts,parts=text_score(ocr,r)
   if ts<35.:continue
   rec={"row":r,"block_index":bi,"ocr_text":ocr,"crop":crop,"text_score":ts,"text_parts":parts}
   if r.id not in candidate_map or ts>candidate_map[r.id]["text_score"]:candidate_map[r.id]=rec
 candidates=sorted(candidate_map.values(),key=lambda q:(q["text_score"],q["row"].priority),reverse=True)[:3];matches=[];diagnostics=[]
 for c in candidates:
  r=c["row"];sh,orb,vs=visual_components(r.reference_image,c["crop"]);dist=None;gps_ok=True
  if r.use_location:
   if lat is None or lon is None or r.latitude is None or r.longitude is None:gps_ok=False
   else:dist=hav(float(lat),float(lon),r.latitude,r.longitude);gps_ok=dist<=r.radius_m
  visual_ok=vs>=float(r.visual_threshold or 55);passed=visual_ok and gps_ok;diag={"id":r.id,"registration_name":r.normalized_fields()["registration_name"],"display_name":r.display_name,"ocr_text":c["ocr_text"],"text_score":c["text_score"],"text_parts":c["text_parts"],"shape_score":sh,"orb_score":orb,"visual_score":vs,"visual_threshold":float(r.visual_threshold or 55),"gps_ok":gps_ok,"distance_m":round(dist,1) if dist is not None else None,"passed":passed};diagnostics.append(diag)
  if passed:
   d=r.public_dict();d.update(diag);d["score"]=round(c["text_score"]*4+vs*6+r.priority*100,1);matches.append(d)
 matches.sort(key=lambda z:z["score"],reverse=True);diagnostics.sort(key=lambda z:(1 if z["passed"] else 0,z["text_score"]+z["visual_score"]),reverse=True);return jsonify(matches=matches,diagnostics=diagnostics,version=VERSION)
def migrate_columns():
 insp=inspect(db.engine)
 if "links" not in insp.get_table_names():return
 cols={c["name"] for c in insp.get_columns("links")};adds=[("registration_name","VARCHAR(300)"),("major_category","VARCHAR(300)"),("minor_category","VARCHAR(500)"),("recognition_text","VARCHAR(1000)")]
 with db.engine.begin() as conn:
  for name,typ in adds:
   if name not in cols:conn.execute(sql_text(f"ALTER TABLE links ADD COLUMN {name} {typ}"))
  conn.execute(sql_text("UPDATE links SET registration_name=COALESCE(NULLIF(registration_name,''),display_name,key_text), major_category=COALESCE(NULLIF(major_category,''),key_text), recognition_text=COALESCE(NULLIF(recognition_text,''),key_text), minor_category=COALESCE(minor_category,'')"))
with app.app_context():db.create_all();migrate_columns()
if __name__=="__main__":app.run(host="0.0.0.0",port=int(os.getenv("PORT","8788")),debug=False)
