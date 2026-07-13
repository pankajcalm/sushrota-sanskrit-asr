"""GRETIL -> per-śāstra Devanagari lexicons. Path->śāstra tag, strip HTML, drop English
header, IAST->Devanagari, tokenize, count. Feeds the śāstra-selector suggestion boosting."""
import os, re, glob, json, html, unicodedata, collections, pickle
from indic_transliteration import sanscript
ROOT='/home/ece/BigDisk/Prathosh/ASR'; SRC=f'{ROOT}/data/gretil/ext/1_sanskr'
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s); return re.sub(r'\s+',' ',s).strip()
DIAC=set('āĀīĪūŪṛṚṝḹḷḸṃṂḥḤśŚṣṢṭṬḍḌṇṆñÑṅṄ')
RULES=[('1_veda/4_upa','upanishad'),('1_veda/5_vedang','vedanga'),('1_veda','veda'),
 ('2_epic','epics'),('3_purana','purana'),
 ('4_rellit/vaisn','vaishnava'),('4_rellit/saiva','saiva'),('4_rellit/buddh','bauddha'),('4_rellit/jaina','jaina'),
 ('5_poetry/1_alam','alankara'),('5_poetry/2_kavya','kavya'),('5_poetry/1_natya','nataka'),('5_poetry/3_drama','nataka'),
 ('5_poetry/1_chandas','chandas'),('5_poetry/5_subhas','subhashita'),('5_poetry/4_narr','kavya'),('5_poetry/6_hist','history'),
 ('6_sastra/1_gram','vyakarana'),('6_sastra/2_lex','kosha'),
 ('6_sastra/3_phil/nyaya','nyaya'),('6_sastra/3_phil/vaisesik','nyaya'),('6_sastra/3_phil/vedanta','vedanta'),
 ('6_sastra/3_phil/advaita','vedanta'),('6_sastra/3_phil/mimamsa','mimamsa'),('6_sastra/3_phil/samkhya','samkhya'),
 ('6_sastra/3_phil/yoga','yoga'),('6_sastra/3_phil/saiva','saiva'),('6_sastra/3_phil/buddh','bauddha'),
 ('6_sastra/4_dharma','dharmashastra'),('6_sastra/5_artha','arthashastra'),('6_sastra/6_kama','kama'),
 ('6_sastra/7_ayur','ayurveda'),('6_sastra/8_jyot','jyotisa'),('6_sastra/9_misc','misc')]
def sastra(rel):
    for pre,tag in RULES:
        if rel.startswith(pre): return tag
    return None
def body(text):
    lines=text.split('\n'); start=0
    for i,l in enumerate(lines):
        if sum(1 for ch in l if ch in DIAC)>=2: start=i; break
    return '\n'.join(lines[start:])
lex=collections.defaultdict(collections.Counter); nfiles=collections.Counter()
files=[f for f in glob.glob(f'{SRC}/**/*.htm', recursive=True) if '/tei/' not in f and '7_fromindonesia' not in f]
print(f'processing {len(files)} htm files', flush=True)
for n,f in enumerate(files):
    rel=os.path.relpath(f, SRC); tag=sastra(rel)
    if not tag: continue
    try: raw=open(f, encoding='utf-8', errors='ignore').read()
    except: continue
    txt=html.unescape(re.sub(r'<[^>]+>',' ', raw))
    try: dev=sanscript.transliterate(body(txt), sanscript.IAST, sanscript.DEVANAGARI)
    except: continue
    for w in norm(dev).split():
        if len(w)>=2: lex[tag][w]+=1
    nfiles[tag]+=1
    if n%300==0: print(f'  {n}/{len(files)}', flush=True)
os.makedirs(f'{ROOT}/data/gretil/lex', exist_ok=True)
pickle.dump({k:dict(v) for k,v in lex.items()}, open(f'{ROOT}/data/gretil/lex/gretil_by_sastra.pkl','wb'))
print('DONE per-śāstra word/type counts:')
for tag in sorted(lex, key=lambda t:-sum(lex[t].values())):
    print(f'  {tag:14} {nfiles[tag]:3} files  {sum(lex[tag].values()):>9} words  {len(lex[tag]):>7} types')
