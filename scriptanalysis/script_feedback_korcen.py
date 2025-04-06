#!/usr/bin/env python3
# 스크립트 피드백버전.py
# 발표 대본 자동 평가 시스템
# korcen 모듈 + 사용자 정의 비속어 목록을 함께 사용
# (방법 A: korcen은 명사 기반, 사용자 정의 사전은 문장 전체 substring 검색)

import sys
sys.path.append('/mnt/d/script_temp/TUK-Graduation-Work/scriptanalysis/script.py')

import re
from collections import Counter
from konlpy.tag import Okt, Kkma
import os
from korcen import korcen

# === 사용자 대본 파일 경로 설정 ===
base_path = r"/mnt/d/script_temp/TUK-Graduation-Work"
file_name = "output_transcription.txt"
file_path = os.path.join(base_path, file_name)

# 사용자 정의 비속어 목록 파일
custom_badwords_file = "custom_profanities.txt"

# 사용자 입력: 발표 시간 (분 단위)
speech_minutes = int(input("발표 시간을 분 단위로 입력하세요: "))

# 사용자 정의 비속어 목록 불러오기
custom_badwords = set()
try:
    with open(custom_badwords_file, "r", encoding="utf-8") as cf:
        for line in cf:
            w = line.strip()
            if w:
                custom_badwords.add(w)
    print(f"[INFO] 사용자 정의 비속어 {len(custom_badwords)}개 로드 완료.")
except FileNotFoundError:
    print(f"[WARN] '{custom_badwords_file}' 파일이 없습니다. 사용자 정의 비속어는 적용되지 않습니다.")

with open(file_path, encoding="utf-8") as f:
    text = f.read()

sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip()]
tagger = Okt()
kkma = Kkma()

uncertainty_count = 0
non_honorific_count = 0
subject_verb_mismatch_count = 0
profanity_count = 0

all_words = []
word_repeat_counter = Counter()
otas_detected = []

non_honorific_examples = []
uncertainty_examples = []
subject_verb_examples = []
profanity_examples = []

# === 추측 표현 감지 ===
def contains_uncertainty_by_morph(pos_tags):
    patterns = [
        ["VA", "EFN"],
        ["ETD", "JX", "VX", "EFN"],
        ["NNB", "JX", "VX", "EFN"],
        ["VV", "XSV", "EPT", "EFN"],
        ["NNG", "XSV", "EP", "EFN"],
        ["NNB", "VV", "EPT"],
    ]
    for i in range(len(pos_tags)):
        for pattern in patterns:
            match = [pos_tags[i+j][1] if i+j < len(pos_tags) else None for j in range(len(pattern))]
            if match == pattern:
                return True
    return False

# === 주어-서술어 호응 검사 ===
def check_subject_verb_agreement(sentence):
    pos_tags = tagger.pos(sentence)
    subject_detected = False
    verb_detected = False
    for i in range(len(pos_tags) - 1):
        word, pos = pos_tags[i]
        next_word, next_pos = pos_tags[i+1]
        if pos == "Noun" and next_pos == "Josa" and next_word in ["이", "가", "은", "는", "를", "을"]:
            subject_detected = True
            break
    verbs = [w for w, p in pos_tags if p == "Verb" or p == "Adjective"]
    last_token = sentence.split()[-1] if sentence else ""
    if verbs or any(last_token.endswith(e) for e in ["습니다", "합니다", "입니다", "되었습니다", "제시합니다"]):
        verb_detected = True
    return subject_detected and verb_detected

informal_tags = {"EFN", "EF", "EFI", "EFQ"}
formal_exceptions = {"ㅂ니다", "습니다", "입니다", "하겠습니다", "ㅂ니까"}

# === 문장별 평가 ===
for idx, sentence in enumerate(sentences):
    kkma_pos = kkma.pos(sentence)
    
    # 1) 추측 표현 감지
    if contains_uncertainty_by_morph(kkma_pos):
        uncertainty_count += 1
        uncertainty_examples.append((idx + 1, sentence))
    
    # 2) 높임말 사용(비격식 여부)
    has_informal = any(pos in informal_tags and word not in formal_exceptions for word, pos in kkma_pos)
    if has_informal:
        non_honorific_count += 1
        non_honorific_examples.append((idx + 1, sentence))
    
    # 3) 주어-서술어 호응
    if not check_subject_verb_agreement(sentence):
        subject_verb_mismatch_count += 1
        subject_verb_examples.append((idx + 1, sentence))
    
    # === 4) 비속어 감지
    # korcen.check()는 명사만 인자로 전달
    noun_string = " ".join(kkma.nouns(sentence))
    korcen_result = korcen.check(noun_string)

    # 사용자 정의 비속어는 문장 전체 substring 검색
    custom_result = any(bad in sentence for bad in custom_badwords)

    # korcen_result OR custom_result 둘 중 하나만 True여도 비속어로 간주
    if korcen_result or custom_result:
        profanity_count += 1
        profanity_examples.append((idx + 1, sentence))

    # === 단어 통계 및 오타(자음/모음단독) 탐지
    ignore_set = {
        "하", "해", "해요", "합니다", "하는", "하자", "하군", "하네요",
        "할", "할게", "할게요", "하겠", "할까", "할까요", "했", "했어",
        "했어요", "했습니다", "했던",
        "는", "을", "ㄴ", "수", "적",
        "은", "의", "이", "를"
    }
    words = [w for w, pos in kkma.pos(sentence)
             if pos not in {"Josa", "Punctuation", "Eomi", "EF", "EFN", "EPT", "EC"}
             and w not in ignore_set]
    all_words.extend(words)
    word_repeat_counter.update(words)

    # 자음/모음 단독 사용 탐지
    for i, char in enumerate(sentence):
        if re.match(r'[ㄱ-ㅎㅏ-ㅣ]', char):
            otas_detected.append((idx + 1, char, i, sentence))

# === 대본 길이 평가 ===
actual_chars = len(text)
min_chars = speech_minutes * 270
max_chars = speech_minutes * 320

if actual_chars < min_chars:
    length_feedback = f"대본이 다소 짧습니다. 약 {min_chars - actual_chars}자 정도 추가를 고려해보세요."
elif actual_chars > max_chars:
    length_feedback = f"대본이 다소 깁니다. 약 {actual_chars - max_chars}자 정도 줄이는 것이 좋습니다."
else:
    length_feedback = "대본 길이가 적절합니다."

# === 결과 출력 ===
print("\n==== 발표 대본 평가 결과 ====")

# 높임말 사용
if non_honorific_examples:
    print("- 높임말 사용: 비격식 문장이 일부 감지되었습니다:")
    for num, s in non_honorific_examples:
        print(f"  [문장 {num}] {s}")
else:
    print("- 높임말 사용: 모든 문장이 격식을 잘 갖추고 있습니다.")

# 추측 표현
if uncertainty_examples:
    print("- 추측 표현: 추측 표현이 포함된 문장이 발견되었습니다:")
    for num, s in uncertainty_examples:
        print(f"  [문장 {num}] {s}")
else:
    print("- 추측 표현: 추측 표현은 발견되지 않았습니다.")

# 주어-서술어 호응
if subject_verb_examples:
    print("- 주어-서술어 호응: 호응이 부족한 문장이 발견되었습니다:")
    for num, s in subject_verb_examples:
        print(f"  [문장 {num}] {s}")
else:
    print("- 주어-서술어 호응: 모든 문장이 주어와 서술어가 적절히 호응합니다.")

# 비속어
if profanity_examples:
    print("- 비속어: 비속어가 포함된 문장이 감지되었습니다:")
    for num, s in profanity_examples:
        print(f"  [문장 {num}] {s}")
else:
    print("- 비속어: 비속어는 전혀 발견되지 않았습니다.")

# 발표 시간 대비 대본 분량
print("\n▶ 발표 시간 대비 대본 분량 평가:")
print(f"  - 발표 시간: {speech_minutes}분")
print(f"  - 대본 길이: {actual_chars}자")
print(f"  - 권장 범위: {min_chars}자 ~ {max_chars}자")
print(f"  {length_feedback}")
