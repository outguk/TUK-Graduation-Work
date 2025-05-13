#!/usr/bin/env python3
# 스크립트 피드백버전.py
# 발표 대본 자동 평가 시스템
# korcen 모듈 + 사용자 정의 비속어 목록을 함께 사용
# (방법 A: korcen은 명사 기반, 사용자 정의 사전은 문장 전체 substring 검색)

import re
import os
from collections import Counter
from konlpy.tag import Okt, Kkma
from korcen import korcen

def load_script(file_path):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip()]
    return text, sentences

def load_custom_badwords(file_path):
    custom_badwords = set()
    try:
        with open(file_path, "r", encoding="utf-8") as cf:
            for line in cf:
                w = line.strip()
                if w:
                    custom_badwords.add(w)
        print(f"[INFO] 사용자 정의 비속어 {len(custom_badwords)}개 로드 완료.")
    except FileNotFoundError:
        print(f"[WARN] '{file_path}' 파일이 없습니다. 사용자 정의 비속어는 적용되지 않습니다.")
    return custom_badwords

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

def check_subject_verb_agreement(sentence, tagger):
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

def analyze_script(sentences, tagger, kkma, custom_badwords):
    informal_tags = {"EFN", "EF", "EFI", "EFQ"}
    formal_exceptions = {"ㅂ니다", "습니다", "입니다", "하겠습니다", "ㅂ니까"}
    ignore_set = {"하", "해", "해요", "합니다", "하는", "하자", "하군", "하네요",
                  "할", "할게", "할게요", "하겠", "할까", "할까요", "했", "했어",
                  "했어요", "했습니다", "했던", "는", "을", "ㄴ", "수", "적",
                  "은", "의", "이", "를"}

    stats = {
        "uncertainty_count": 0,
        "non_honorific_count": 0,
        "subject_verb_mismatch_count": 0,
        "profanity_count": 0,
        "uncertainty_examples": [],
        "non_honorific_examples": [],
        "subject_verb_examples": [],
        "profanity_examples": [],
        "otas_detected": [],
        "word_repeat_counter": Counter(),
        "all_words": []
    }

    for idx, sentence in enumerate(sentences):
        kkma_pos = kkma.pos(sentence)

        if contains_uncertainty_by_morph(kkma_pos):
            stats["uncertainty_count"] += 1
            stats["uncertainty_examples"].append((idx + 1, sentence))

        has_informal = any(pos in informal_tags and word not in formal_exceptions for word, pos in kkma_pos)
        if has_informal:
            stats["non_honorific_count"] += 1
            stats["non_honorific_examples"].append((idx + 1, sentence))

        if not check_subject_verb_agreement(sentence, tagger):
            stats["subject_verb_mismatch_count"] += 1
            stats["subject_verb_examples"].append((idx + 1, sentence))

        noun_string = " ".join(kkma.nouns(sentence))
        korcen_result = korcen.check(noun_string)
        custom_result = any(bad in sentence for bad in custom_badwords)
        if korcen_result or custom_result:
            stats["profanity_count"] += 1
            stats["profanity_examples"].append((idx + 1, sentence))

        words = [w for w, pos in kkma.pos(sentence)
                 if pos not in {"Josa", "Punctuation", "Eomi", "EF", "EFN", "EPT", "EC"}
                 and w not in ignore_set]
        stats["all_words"].extend(words)
        stats["word_repeat_counter"].update(words)

        for i, char in enumerate(sentence):
            if re.match(r'[ㄱ-ㅎㅏ-ㅣ]', char):
                stats["otas_detected"].append((idx + 1, char, i, sentence))

    return stats

def evaluate_length(text, speech_minutes):
    actual_chars = len(text)
    min_chars = speech_minutes * 270
    max_chars = speech_minutes * 320

    if actual_chars < min_chars:
        feedback = f"대본이 다소 짧습니다. 약 {min_chars - actual_chars}자 정도 추가를 고려해보세요."
    elif actual_chars > max_chars:
        feedback = f"대본이 다소 깁니다. 약 {actual_chars - max_chars}자 정도 줄이는 것이 좋습니다."
    else:
        feedback = "대본 길이가 적절합니다."

    return actual_chars, min_chars, max_chars, feedback

def print_results(stats, speech_minutes, actual_chars, min_chars, max_chars, length_feedback):
    print("\n==== 발표 대본 평가 결과 ====")

    if stats["non_honorific_examples"]:
        print("- 높임말 사용: 비격식 문장이 일부 감지되었습니다:")
        for num, s in stats["non_honorific_examples"]:
            print(f"  [문장 {num}] {s}")
    else:
        print("- 높임말 사용: 모든 문장이 격식을 잘 갖추고 있습니다.")

    if stats["uncertainty_examples"]:
        print("- 추측 표현: 추측 표현이 포함된 문장이 발견되었습니다:")
        for num, s in stats["uncertainty_examples"]:
            print(f"  [문장 {num}] {s}")
    else:
        print("- 추측 표현: 추측 표현은 발견되지 않았습니다.")

    if stats["subject_verb_examples"]:
        print("- 주어-서술어 호응: 호응이 부족한 문장이 발견되었습니다:")
        for num, s in stats["subject_verb_examples"]:
            print(f"  [문장 {num}] {s}")
    else:
        print("- 주어-서술어 호응: 모든 문장이 주어와 서술어가 적절히 호응합니다.")

    if stats["profanity_examples"]:
        print("- 비속어: 비속어가 포함된 문장이 감지되었습니다:")
        for num, s in stats["profanity_examples"]:
            print(f"  [문장 {num}] {s}")
    else:
        print("- 비속어: 비속어는 전혀 발견되지 않았습니다.")

    print("\n▶ 발표 시간 대비 대본 분량 평가:")
    print(f"  - 발표 시간: {speech_minutes}분")
    print(f"  - 대본 길이: {actual_chars}자")
    print(f"  - 권장 범위: {min_chars}자 ~ {max_chars}자")
    print(f"  {length_feedback}")

def main():
    speech_minutes = int(input("발표 시간을 분 단위로 입력하세요: "))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_name = "output_transcription.txt"
    custom_badwords_file = "custom_profanities.txt"

    file_path = os.path.join(BASE_DIR, file_name)
    custom_path = os.path.join(BASE_DIR, custom_badwords_file)

    text, sentences = load_script(file_path)
    custom_badwords = load_custom_badwords(custom_path)

    tagger = Okt()
    kkma = Kkma()

    stats = analyze_script(sentences, tagger, kkma, custom_badwords)
    actual_chars, min_chars, max_chars, length_feedback = evaluate_length(text, speech_minutes)
    print_results(stats, speech_minutes, actual_chars, min_chars, max_chars, length_feedback)

if __name__ == "__main__":
    main()
