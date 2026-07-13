"""GRETIL -> per-śāstra Devanagari lexicons, v2: per-file scheme DETECTION + validity filter."""
import os, re, glob, json, html, unicodedata, collections, pickle
from indic_transliteration import sanscript
from indic_transliteration.detect import detect as detect_scheme
ROOT='/home/ece/BigDisk/Prathosh/ASR'; SRC=f'{ROOT}/data/gretil/ext/1_sanskr'
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s); return re.sub(r'\s+',' ',s).strip()
SMAP={'IAST':sanscript.IAST,'HK':sanscript.HK,'Velthuis':sanscript.VELTHUIS,'SLP1':sanscript.SLP1,'ITRANS':sanscript.ITRANS,'Kolkata':sanscript.IAST}
def to_dev(body):
    try: sch=detect_scheme(body[:3000])
    except: sch='IAST'
    if sch=='Devanagari': return body
    sc=SMAP.get(sch, sanscript.IAST)
    try: return sanscript.transliterate(body, sc, sanscript.DEVANAGARI)
    except: return ''
CONS=set('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')
def valid(w):
    return len(w)>=3 and any(c in CONS for c in w) and (w.count('्')/len(w))<=0.4
RULES=[('1_veda/4_upa','upanishad'),('1_veda/5_vedang','vedanga'),('1_veda','veda'),('2_epic','epics'),('3_purana','purana'),
 ('4_rellit/vaisn','vaishnava'),('4_rellit/saiva','saiva'),('4_rellit/buddh','bauddha'),('4_rellit/jaina','jaina'),
 ('5_poetry/1_alam','alankara'),('5_poetry/2_kavya','kavya'),('5_poetry/1_natya','nataka'),('5_poetry/3_drama','nataka'),
 ('5_poetry/1_chandas','chandas'),('5_poetry/5_subhas','subhashita'),('5_poetry/4_narr','kavya'),('5_poetry/6_hist','history'),
 ('6_sastra/1_gram','vyakarana'),('6_sastra/2_lex','kosha'),('6_sastra/3_phil/nyaya','nyaya'),('6_sastra/3_phil/vaisesik','nyaya'),
 ('6_sastra/3_phil/vedanta','vedanta'),('6_sastra/3_phil/advaita','vedanta'),('6_sastra/3_phil/mimamsa','mimamsa'),
 ('6_sastra/3_phil/samkhya','samkhya'),('6_sastra/3_phil/yoga','yoga'),('6_sastra/3_phil/saiva','saiva'),('6_sastra/3_phil/buddh','bauddha'),
 ('6_sastra/4_dharma','dharmashastra'),('6_sastra/5_artha','arthashastra'),('6_sastra/6_kama','kama'),('6_sastra/7_ayur','ayurveda'),
 ('6_sastra/8_jyot','jyotisa'),('6_sastra/9_misc','misc')]
def sastra(rel):
    for pre,tag in RULES:
        if rel.startswith(pre): return tag
    return None
DIAC=set('āĀīĪūŪṛṚṝḹḷḸṃṂḥḤśŚṣṢṭṬḍḌṇṆñÑṅṄ')
def body(text):
    lines=text.split('\n'); start=0
    for i,l in enumerate(lines):
        if sum(1 for ch in l if ch in DIAC)>=2 or sum(1 for ch in l if 'ऀ'<=ch<='ॿ')>=4: start=i; break
    return '\n'.join(lines[start:])
lex=collections.defaultdict(collections.Counter); nf=collections.Counter()
files=[f for f in glob.glob(f'{SRC}/**/*.htm', recursive=True) if '/tei/' not in f and '7_fromindonesia' not in f]
print(f'processing {len(files)} files', flush=True)
for n,f in enumerate(files):
    rel=os.path.relpath(f,SRC); tag=sastra(rel)
    if not tag: continue
    try: raw=open(f,encoding='utf-8',errors='ignore').read()
    except: continue
    txt=html.unescape(re.sub(r'<[^>]+>',' ',raw))
    dev=to_dev(body(txt))
    ws=[w for w in norm(dev).split() if valid(w)]
    if ws: lex[tag].update(ws); nf[tag]+=1
    if n%300==0: print(f'  {n}/{len(files)}', flush=True)
pickle.dump({k:dict(v) for k,v in lex.items()}, open(f'{ROOT}/data/gretil/lex/gretil_by_sastra.pkl','wb'))
print('DONE (v2, scheme-detect + validity):')
for tag in sorted(lex, key=lambda t:-sum(lex[t].values())):
    top=' '.join(w for w,_ in collections.Counter(lex[tag]).most_common(8))
    print(f'  {tag:13} {nf[tag]:3}f {len(lex[tag]):>7}types | {top}')
