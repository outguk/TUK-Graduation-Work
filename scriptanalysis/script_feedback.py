# 스크립트 피드백버전.py
# 발표 대본 자동 평가 시스템
# 이 코드는 한국어 발표 대본(STT 텍스트 등)을 문장 단위로 분석하여,
# 언어학적 기준에 따라 품질을 평가하고 구체적인 피드백을 제공합니다.
# 평가 항목: 높임말 사용, 추측 표현 여부, 주어-서술어 호응 관계,
#          발표 시간 대비 적절한 분량, 비속어 포함 여부
# 추가 기능: 자음/모음 단독 사용(오타 가능성) 탐지,
#          문제 문장에 대한 구체적 피드백 표시

import sys
sys.path.append('/mnt/d/script_temp/TUK-Graduation-Work/scriptanalysis/script.py')

import re
from collections import Counter
from konlpy.tag import Okt, Kkma
import os
from badword_check import BadWord

# === 사용자 대본 파일 경로 설정 ===
base_path = r"/mnt/d/script_temp/TUK-Graduation-Work"
file_name = "output_transcription.txt"
file_path = os.path.join(base_path, file_name)

# 사용자 입력: 발표 시간 (분 단위)
speech_minutes = int(input("발표 시간을 분 단위로 입력하세요: "))

with open(file_path, encoding="utf-8") as f:
    text = f.read()

sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip()]
tagger = Okt()
kkma = Kkma()
badword_model = BadWord.load_badword_model()

uncertainty_count = 0
non_honorific_count = 0
subject_verb_mismatch_count = 0
profanity_count = 0

all_words = []
word_repeat_counter = Counter()  # (반복 단어 분석은 더 이상 출력하지 않음)
otas_detected = []

non_honorific_examples = []
uncertainty_examples = []
subject_verb_examples = []
profanity_examples = []

# 정제된 추측 표현 패턴 (정확도 높은 조합 위주)
def contains_uncertainty_by_morph(pos_tags):
    patterns = [
        ["VA", "EFN"],                    # ~같습니다
        ["ETD", "JX", "VX", "EFN"],       # ~일지도 모릅니다
        ["NNB", "JX", "VX", "EFN"],       # ~수도 있습니다
        ["VV", "XSV", "EPT", "EFN"],      # ~보일 수 있습니다
        ["NNG", "XSV", "EP", "EFN"],      # ~생각됩니다
        ["NNB", "VV", "EPT"],             # ~할 수 있다 (보완 필요)
    ]
    for i in range(len(pos_tags)):
        for pattern in patterns:
            match = [pos_tags[i + j][1] if i + j < len(pos_tags) else None for j in range(len(pattern))]
            if match == pattern:
                return True
    return False

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

for idx, sentence in enumerate(sentences):
    kkma_pos = kkma.pos(sentence)
    if contains_uncertainty_by_morph(kkma_pos):
        uncertainty_count += 1
        uncertainty_examples.append((idx + 1, sentence))
    
    has_informal = any(pos in informal_tags and word not in formal_exceptions for word, pos in kkma_pos)
    if has_informal:
        non_honorific_count += 1
        non_honorific_examples.append((idx + 1, sentence))
    
    if not check_subject_verb_agreement(sentence):
        subject_verb_mismatch_count += 1
        subject_verb_examples.append((idx + 1, sentence))
    
    # 비속어 감지 (임계값 설정 최대 1)
    threshold = 0.4
    data = BadWord.preprocessing(sentence)
    prediction = badword_model.predict(data)
    if prediction >= threshold:
        profanity_count += 1
        profanity_examples.append((idx + 1, sentence))
    
    # 단어 추출: Kkma 태깅 결과를 이용하여 기본 필터링 후,
    # 추가로 ignore_set에 포함된 불필요한 단어들을 제거합니다.
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
    
    for i, char in enumerate(sentence):
        if re.match(r'[ㄱ-ㅎㅏ-ㅣ]', char):
            otas_detected.append((idx + 1, char, i, sentence))

# -----------------------
# 대본 길이 평가 (발표 시간 대비 적정 분량 판단)
actual_chars = len(text)
min_chars = speech_minutes * 270
max_chars = speech_minutes * 320

if actual_chars < min_chars:
    length_feedback = f"대본이 다소 짧습니다. 약 {min_chars - actual_chars}자 정도 추가를 고려해보세요."
elif actual_chars > max_chars:
    length_feedback = f"대본이 다소 깁니다. 약 {actual_chars - max_chars}자 정도 줄이는 것이 좋습니다."
else:
    length_feedback = "대본 길이가 적절합니다."

# -----------------------
# 출력: 점수 대신 피드백 방식
print("\n==== 발표 대본 평가 결과 ====")

# 높임말 사용 피드백
if non_honorific_examples:
    print("- 높임말 사용: 비격식 문장이 일부 감지되었습니다:")
    for num, s in non_honorific_examples:
        print(f"  [문장 {num}] {s}")
else:
    print("- 높임말 사용: 모든 문장이 격식을 잘 갖추고 있습니다.")

# 추측 표현 피드백
if uncertainty_examples:
    print("- 추측 표현: 추측 표현이 포함된 문장이 발견되었습니다:")
    for num, s in uncertainty_examples:
        print(f"  [문장 {num}] {s}")
else:
    print("- 추측 표현: 추측 표현은 발견되지 않았습니다.")

# 주어-서술어 호응 피드백
if subject_verb_examples:
    print("- 주어-서술어 호응: 호응이 부족한 문장이 발견되었습니다:")
    for num, s in subject_verb_examples:
        print(f"  [문장 {num}] {s}")
else:
    print("- 주어-서술어 호응: 모든 문장이 주어와 서술어가 적절히 호응합니다.")

# 비속어 피드백
if profanity_examples:
    print("- 비속어: 비속어가 포함된 문장이 감지되었습니다:")
    for num, s in profanity_examples:
        print(f"  [문장 {num}] {s}")
else:
    print("- 비속어: 비속어는 전혀 발견되지 않았습니다.")

# 대본 길이 피드백
print("\n▶ 발표 시간 대비 대본 분량 평가:")
print(f"  - 발표 시간: {speech_minutes}분")
print(f"  - 대본 길이: {actual_chars}자")
print(f"  - 권장 범위: {min_chars}자 ~ {max_chars}자")
print(f"  {length_feedback}")
