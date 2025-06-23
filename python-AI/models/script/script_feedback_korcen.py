#!/usr/bin/env python3
# 발표 대본 자동 평가 모듈 (KorCen + 사용자 정의 비속어)
# 2025-06 패치 ─ ‘개인·개선’ 오검지정 및 custom 사전 충돌 수정

from __future__ import annotations
import os, re
from collections import Counter
from typing import Any, Dict, List, Tuple

from konlpy.tag import Okt, Kkma
from korcen import korcen

# ───────────────────────────────
# 0. 종결 격식 · 높임 호응 (변경 없음)
# ───────────────────────────────
FORMAL_SIGHA = re.compile(r"(?:ㅂ니다|습니다|입니다|입니까|습니까|시죠|십시오|겠습니다)$")
FORMAL_HAEO  = re.compile(r"(?:해요|아요|어요|세요|죠|지요|겠죠|이죠)$")

def is_formal_polite(pos: List[Tuple[str,str]], sent: str) -> bool:
    for w,t in reversed(pos):
        if t.startswith("E"):
            ending=w; break
    else:
        ending=sent.strip()[-10:]
    ending=re.sub(r"[.…!?]+$","",ending)
    return bool(FORMAL_SIGHA.search(ending) or FORMAL_HAEO.search(ending))

HONORIFIC_OK_PHRASES=("주시기바랍니다","주십시오","주셔서감사합니다")
def _has_predicate(pos): return any(t.startswith(("VV","VA","VCP","VCN")) for _,t in pos)\
                            or any(t.startswith("EF") for _,t in pos)
def is_sentence_fragment(s,p): return (not _has_predicate(p)) or p[-1][1].startswith("EC")

def check_honorific_agreement(pos)->bool:
    full="".join(w for w,_ in pos)
    if any(full.endswith(ph) for ph in HONORIFIC_OK_PHRASES): return True
    subj_honor=subj_self=verb_honor=False
    for i,(w,t) in enumerate(pos):
        if t=="JKS" and i:
            noun=pos[i-1][0]
            subj_self  = noun in {"저","나","우리"}
            subj_honor = noun.endswith("님") or noun=="분" or w=="께서"
        if t.startswith(("EP","EF")) and "시" in w: verb_honor=True
    if subj_honor and not verb_honor: return False
    if verb_honor and (subj_self or not subj_honor): return False
    return True

# ───────────────────────────────
# 0-C. 인사·감사 화이트리스트
# ───────────────────────────────
_GREETING_END_RE=re.compile(r"(?:안녕하(?:세요|십니까|신가요))\s*[.!?]?$")
_THANKS_END_RE  =re.compile(r"(?:감사(?:합니다|드립니다)|고맙습니다|수고하셨습니다)\s*[.!?]?$")
def is_greeting_or_thanks(s:str)->bool:
    s=s.strip()
    return bool(_GREETING_END_RE.match(s) or _THANKS_END_RE.search(s))

# ───────────────────────────────
# 1. 문장 분리
# ───────────────────────────────
SENT_SPLIT_RE=re.compile(r"(?<!\d)[.!?](?!\d)\s+|\n+")

def load_script(fp:str): 
    text=open(fp,encoding="utf-8").read()
    s=[t.strip() for t in SENT_SPLIT_RE.split(text) if t.strip()]
    return text,s

def load_custom_badwords(fp:str)->set[str]:
    if fp and os.path.exists(fp):
        with open(fp,encoding="utf-8") as f:
            bad={ln.strip() for ln in f if ln.strip()}
        # 〈중요〉 단독 ‘개’는 슬랭 전용 로직으로 다루므로 삭제
        bad.discard("개")
        print(f"[INFO] 사용자 정의 비속어 {len(bad)}개 로드 (단독 '개' 제외)")
        return bad
    return set()

# ───────────────────────────────
# 2-A. 불확실 표현 (생략‧동일)
# ───────────────────────────────
UNCERTAINTY_LEXICAL={"아마","아마도","어쩌면","혹시","추정","추측","예상","가정","미정","불확실","모르","모름"}
UNCERTAINTY_REGEX=re.compile(r"(?:것\s*같[다아요]|듯\s*하[다아요]|듯싶[다어요]|보인[다아요]|추정된다|예상된다|가능성이\s*(?:있|높|낮)(?:[^\w]|$))")
GUESS_ADVERB_W_GESS=re.compile(r"(아마|아마도|어쩌면|혹시)[^\n]{0,30}?겠")
UNCERTAINTY_OK_PHRASES=("주시기 바랍니다","주십시오","주셔서 감사합니다")

def contains_uncertainty(s:str,pos)->bool:
    if any(p in s for p in UNCERTAINTY_OK_PHRASES): return False
    if re.search(r"\b수\s+(?:있|없)[^\s]*\s*(?:었|었습|습|다|어요|습니다)\b",s): return False
    if "고자 합니다" in s or re.search(r"\b[하듣보읽]겠습니다\b",s): return False
    if UNCERTAINTY_REGEX.search(s) or GUESS_ADVERB_W_GESS.search(s): return True
    if any(w in UNCERTAINTY_LEXICAL for w,_ in pos): return True
    return any(t=="EPH" for _,t in pos)

# ───────────────────────────────
# 2-B. 비속어 검사  ★ PATCHED ★
# ───────────────────────────────
SLANG_PREFIX=re.compile(
    r"\b개\s*[-]?\s*(?!인[가-힣]*|발[가-힣]*|선[가-힣]*|성[가-힣]*|방[가-힣]*|념[가-힣]*|별[가-힣]*|체[가-힣]*|월[가-힣]*\b)"
    r"[가-힣]{1,}\b"
)
GAE_PREFIX_WHITELIST=("개인","개발","개선","개성","개방","개별","개체","개념","개월")

def contains_profanity(sent:str, extra_bad:set[str])->bool:
    # ① ‘개+X’ 은어 (띄어쓰기·하이픈 포함 허용)
    if SLANG_PREFIX.search(sent):
        return True

    # ② 사용자 정의 금칙어
    for bad in extra_bad:
        # 길이 1~2 글자는 완전일치만(‘개’는 이미 제거)
        if len(bad)<=2:
            if re.search(rf"\b{re.escape(bad)}\b",sent): return True
        else:
            if re.search(rf"\b{re.escape(bad)}[가-힣]*\b",sent): return True

    # ③ KorCen + 접두 화이트리스트
    for token in re.findall(r"[가-힣]+",sent):
        if any(token.startswith(pref) for pref in GAE_PREFIX_WHITELIST):
            continue       # ‘개인/개선…’→ 통과
        if token=="개":     # 단독 '개'는 욕 아님
            continue
        if korcen.check(token):
            return True
    return False

# ───────────────────────────────
# 3. 분석 루프
# ───────────────────────────────
def analyze_script(sents:List[str], kk:Kkma, bad:set[str])->Dict[str,Any]:
    ignore={"하","해","해요","합니다","하는","하자","하군","하네요","할","할게",
            "할게요","하겠","할까","할까요","했","했어","했어요","했습니다","했던",
            "는","을","ㄴ","수","적","은","의","이","를"}
    st=dict(uncertainty_count=0,non_honorific_count=0,subject_verb_mismatch_count=0,profanity_count=0,
            uncertainty_examples=[],non_honorific_examples=[],subject_verb_examples=[],profanity_examples=[],
            otas_detected=[],word_repeat_counter=Counter(),all_words=[])
    for idx,s in enumerate(sents):
        pos=kk.pos(s)
        if contains_uncertainty(s,pos):
            st["uncertainty_count"]+=1; st["uncertainty_examples"].append((idx+1,s))
        if not is_formal_polite(pos,s):
            st["non_honorific_count"]+=1; st["non_honorific_examples"].append((idx+1,s))
        if not is_greeting_or_thanks(s):
            if is_sentence_fragment(s,pos) or not check_honorific_agreement(pos):
                st["subject_verb_mismatch_count"]+=1; st["subject_verb_examples"].append((idx+1,s))
        if contains_profanity(s,bad):
            st["profanity_count"]+=1; st["profanity_examples"].append((idx+1,s))
        words=[w for w,t in pos if t not in {"Josa","Punctuation","Eomi","EF","EFN","EPT","EC"} and w not in ignore]
        st["all_words"].extend(words); st["word_repeat_counter"].update(words)
    return st

# ───────────────────────────────
# 4. 길이 평가 & 래퍼 
# ───────────────────────────────
def evaluate_length(txt:str,minu:int)->Tuple[int,int,int,str]:
    n=len(txt); lo,hi=minu*270,minu*320
    if n<lo: fb=f"대본이 다소 짧습니다. 약 {lo-n}자 정도 추가를 고려해보세요."
    elif n>hi: fb=f"대본이 다소 깁니다. 약 {n-hi}자 정도 줄이는 것이 좋습니다."
    else: fb="대본 길이가 적절합니다."
    return n,lo,hi,fb

def run_script_feedback(script_path:str|None=None,script_text:str|None=None,
                        speech_minutes:int=1,custom_badwords_path:str|None=None)->Dict[str,Any]:
    if script_path:
        txt,sents=load_script(script_path)
    elif script_text:
        txt=script_text.strip(); sents=[t.strip() for t in SENT_SPLIT_RE.split(txt) if t.strip()]
    else:
        raise ValueError("script_path 또는 script_text 중 하나는 필수입니다.")
    bad=load_custom_badwords(custom_badwords_path)
    kk=Kkma()
    st=analyze_script(sents,kk,bad)
    n,lo,hi,fb=evaluate_length(txt,speech_minutes)
    return {"length":n,"min_length":lo,"max_length":hi,"length_feedback":fb,**st}
