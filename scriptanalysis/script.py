# 발표 대본 자동 평가 시스템
# 이 코드는 한국어 발표 대본(STT 텍스트 등)을 문장 단위로 분석하여, 언어학적 기준에 따라 품질을 평가하고 피드백을 제공합니다.
# 평가 항목: 높임말 사용, 추측 표현 여부, 어휘 다양성, 주어-서술어 호응 관계, 발표 시간 대비 적절한 분량, 비속어 포함 여부
# 추가 기능: 자음/모음 단독 사용(오타 가능성) 탐지, 반복 단어 빈도 출력, 문제 문장에 대한 구체적 피드백 표시

import sys
sys.path.append('/mnt/d/script_temp/TUK-Graduation-Work/scriptanalysis/script.py')

import re
from collections import Counter
from konlpy.tag import Okt, Kkma
import os
from badword_check import BadWord

# === 사용자 대본 파일 경로 설정 ===
# base_path = r"/Users/junpyo/TUK-Graduation-Work"
base_path = r"/mnt/d/script_temp/TUK-Graduation-Work"

file_name = "output_transcription.txt"
file_path = os.path.join(base_path, file_name)

# 사용자 입력: 발표 시간 (분 단위)
speech_minutes = 5

# 평가 항목별 만점 점수 설정 (총 100점)
criteria = {
    "honorific": 20,
    "uncertainty": 20,
    "vocab_diversity": 20,
    "subject_verb_agreement": 20,
    "length_appropriateness": 10,
    "profanity": 10
}

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
word_repeat_counter = Counter()
otas_detected = []

non_honorific_examples = []
uncertainty_examples = []
subject_verb_examples = []
profanity_examples = []

# 🔍 정제된 추측 표현 패턴 (정확도 높은 조합 위주)
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

    data = BadWord.preprocessing(sentence)
    if badword_model.predict(data) == 1:
        profanity_count += 1
        profanity_examples.append((idx + 1, sentence))

    words = [w for w, pos in tagger.pos(sentence) if pos not in ["Josa", "Punctuation", "Eomi"]]
    all_words.extend(words)
    word_repeat_counter.update(words)

    for i, char in enumerate(sentence):
        if re.match(r'[ㄱ-ㅎㅏ-ㅣ]', char):
            otas_detected.append((idx + 1, char, i, sentence))

# 어휘 다양성
total_word_count = len(all_words)
type_word_count = len(set(all_words))
vocab_diversity_score = type_word_count / total_word_count if total_word_count else 0

# 대본 길이 평가 (발표 시간 대비 적정 분량 판단)
actual_chars = len(text)
min_chars = speech_minutes * 270
max_chars = speech_minutes * 320

if actual_chars < min_chars:
    length_score = criteria["length_appropriateness"] * (actual_chars / min_chars)
elif actual_chars > max_chars:
    over_ratio = (actual_chars - max_chars) / max_chars
    length_score = max(0, criteria["length_appropriateness"] * (1 - over_ratio))
else:
    length_score = criteria["length_appropriateness"]

# 점수 집계
scores = dict()
scores["honorific"] = max(0, criteria["honorific"] - non_honorific_count)
scores["uncertainty"] = max(0, criteria["uncertainty"] - uncertainty_count)
scores["vocab_diversity"] = vocab_diversity_score * criteria["vocab_diversity"]
scores["subject_verb_agreement"] = max(0, criteria["subject_verb_agreement"] - subject_verb_mismatch_count)
scores["length_appropriateness"] = length_score
scores["profanity"] = max(0, criteria["profanity"] - profanity_count)

total_score = sum(scores.values())

# 출력
print("\n==== 발표 대본 평가 결과 ====")
for key in scores:
    print(f"{key.capitalize().replace('_', ' ')} Score: {scores[key]:.1f}/{criteria[key]}")

print(f"\n▶ 최종 점수: {total_score:.1f}/100")

if non_honorific_examples:
    print(f"- 비격식체 문장 수: {len(non_honorific_examples)}개")
    for num, s in non_honorific_examples:
        print(f"  [문장 {num}] 비격식 어미 감지됨: {s}")
if uncertainty_examples:
    print(f"- 추측 표현이 포함된 문장 수: {len(uncertainty_examples)}개")
    for num, s in uncertainty_examples:
        print(f"  [문장 {num}] 추측 표현 포함: {s}")
if vocab_diversity_score < 0.5:
    print(f"- 어휘 다양성이 낮음 (비율: {vocab_diversity_score:.2f})")
if subject_verb_examples:
    print(f"- 주어-서술어 호응이 부족한 문장 수: {len(subject_verb_examples)}개")
    for num, s in subject_verb_examples:
        print(f"  [문장 {num}] 주어-서술어 구조 불완전: {s}")
if profanity_examples:
    print(f"- 비속어 포함 문장 수: {len(profanity_examples)}개")
    for num, s in profanity_examples:
        print(f"  [문장 {num}] 비속어 포함됨: {s}")

print("\n▶ 발표 시간 대비 대본 분량 평가:")
print(f"  - 발표 시간: {speech_minutes}분")
print(f"  - 대본 길이: {actual_chars}자")
print(f"  - 권장 범위: {min_chars}자 ~ {max_chars}자")
if actual_chars < min_chars:
    print(f"  ❗ 대본이 다소 짧습니다. 약 {min_chars - actual_chars}자 정도 추가를 고려해보세요.")
elif actual_chars > max_chars:
    print(f"  ❗ 대본이 다소 깁니다. 약 {actual_chars - max_chars}자 정도 줄이는 것이 좋습니다.")
else:
    print(f"  ✅ 발표 시간에 적절한 분량입니다.")

if otas_detected:
    print("\n▶ 오타 의심 문자 감지됨 (자음/모음 단독 사용):")
    for num, char, pos, line in otas_detected:
        print(f"  - 문장 {num} / 위치 {pos}: '{char}' in \"{line[:30]}...")
    print("→ 자음/모음 단독 사용은 오타일 가능성이 높습니다. 수정해주세요.")

print("\n▶ 자주 반복된 단어:")
for word, count in word_repeat_counter.most_common(5):
    if count > 1:
        print(f"  - {word}: {count}회")