#!/usr/bin/env python3
# 발표 대본 자동 평가 모듈 (KorCen + 사용자 정의 비속어)
# 2025-05 리팩터링 — 경로 하드코딩 제거, 재사용 함수화

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List, Tuple, TypedDict

from konlpy.tag import Okt, Kkma
from korcen import korcen

class ScriptAnalysisStats(TypedDict):
    uncertainty_count: int
    non_honorific_count: int
    subject_verb_mismatch_count: int
    profanity_count: int
    uncertainty_examples: List[Tuple[int, str]]
    non_honorific_examples: List[Tuple[int, str]]
    subject_verb_examples: List[Tuple[int, str]]
    profanity_examples: List[Tuple[int, str]]
    otas_detected: List[Tuple[int, str, int, str]]
    word_repeat_counter: Counter[str]
    all_words: List[str]

# ─────────────────────────────────────────────
# 1. 유틸 함수
# ─────────────────────────────────────────────
def load_script(file_path: str) -> Tuple[str, List[str]]:
    """텍스트 파일을 읽어 문장 리스트와 전체 문자열 반환"""
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
    return text, sentences


def load_custom_badwords(file_path: str) -> set[str]:
    """사용자 정의 비속어 목록 로드(없으면 빈 set)"""
    custom = set()
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            custom.update({line.strip() for line in f if line.strip()})
        print(f"[INFO] 사용자 정의 비속어 {len(custom)}개 로드 완료.")
    else:
        print(f"[WARN] '{file_path}' 파일이 없습니다. 사용자 정의 비속어 미적용.")
    return custom


# ─────────────────────────────────────────────
# 2. 언어 규칙 검사 서브루틴
# ─────────────────────────────────────────────
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
            if [pos_tags[i + j][1] if i + j < len(pos_tags) else None for j in range(len(pattern))] == pattern:
                return True
    return False


def check_subject_verb_agreement(sentence: str, tagger: Okt) -> bool:
    pos_tags = tagger.pos(sentence)
    subject = any(
        p == "Noun" and i + 1 < len(pos_tags) and pos_tags[i + 1][1] == "Josa"
        for i, (_, p) in enumerate(pos_tags)
    )
    verb = any(p in {"Verb", "Adjective"} for _, p in pos_tags) or sentence.endswith(
        ("습니다", "합니다", "입니다", "되었습니다", "제시합니다")
    )
    return subject and verb


# ─────────────────────────────────────────────
# 3. 핵심 분석 함수
# ─────────────────────────────────────────────
def analyze_script(
    sentences: List[str],
    tagger: Okt,
    kkma: Kkma,
    custom_badwords: set[str],
) -> ScriptAnalysisStats:
    informal_tags = {"EFN", "EF", "EFI", "EFQ"}
    formal_exceptions = {"ㅂ니다", "습니다", "입니다", "하겠습니다", "ㅂ니까"}
    ignore_set = {
        "하",
        "해",
        "해요",
        "합니다",
        "하는",
        "하자",
        "하군",
        "하네요",
        "할",
        "할게",
        "할게요",
        "하겠",
        "할까",
        "할까요",
        "했",
        "했어",
        "했어요",
        "했습니다",
        "했던",
        "는",
        "을",
        "ㄴ",
        "수",
        "적",
        "은",
        "의",
        "이",
        "를",
    }

    stats: ScriptAnalysisStats = {
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
        "all_words": [],
    }

    for idx, sentence in enumerate(sentences):
        kkma_pos = kkma.pos(sentence)

        # ① 추측 표현
        if contains_uncertainty_by_morph(kkma_pos):
            stats["uncertainty_count"] += 1
            stats["uncertainty_examples"].append((idx + 1, sentence))

        # ② 높임말
        if any(pos in informal_tags and word not in formal_exceptions for word, pos in kkma_pos):
            stats["non_honorific_count"] += 1
            stats["non_honorific_examples"].append((idx + 1, sentence))

        # ③ 주어-서술어 호응
        if not check_subject_verb_agreement(sentence, tagger):
            stats["subject_verb_mismatch_count"] += 1
            stats["subject_verb_examples"].append((idx + 1, sentence))

        # ④ 비속어
        noun_string = " ".join(kkma.nouns(sentence))
        if korcen.check(noun_string) or any(bad in sentence for bad in custom_badwords):
            stats["profanity_count"] += 1
            stats["profanity_examples"].append((idx + 1, sentence))

        # ⑤ 단어 빈도 & 오타
        words = [
            w
            for w, pos in kkma.pos(sentence)
            if pos not in {"Josa", "Punctuation", "Eomi", "EF", "EFN", "EPT", "EC"} and w not in ignore_set
        ]
        stats["all_words"].extend(words)
        stats["word_repeat_counter"].update(words)
        stats["otas_detected"].extend(
            (idx + 1, ch, i, sentence) for i, ch in enumerate(sentence) if re.match(r"[ㄱ-ㅎㅏ-ㅣ]", ch)
        )
    return stats


# ─────────────────────────────────────────────
# 4. 길이 평가
# ─────────────────────────────────────────────
def evaluate_length(text: str, speech_minutes: int) -> Tuple[int, int, int, str]:
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


# ─────────────────────────────────────────────
# 5. FastAPI용 래퍼 함수
# ─────────────────────────────────────────────
def run_script_feedback(
    script_path: str | None = None,  # 수정: str | None으로 변경, 기본값 None
    script_text: str | None = None,  # 추가: 텍스트 입력 파라미터
    speech_minutes: int = 1,
    custom_badwords_path: str | None = None,
) -> Dict[str, Any]:
    """
    FastAPI 등 외부에서 호출해 MongoDB에 바로 넣을 수 있는 최상위 함수.
    :param script_path: 업로드된 대본 텍스트 파일 절대경로 (선택)
    :param script_text: 직접 입력된 대본 텍스트 (선택)
    :param speech_minutes: 발표 시간(분)
    :param custom_badwords_path: 사용자 정의 비속어 txt 경로 (없으면 None)
    :return: MongoDB 저장용 dict
    """
    if script_path:
        text, sentences = load_script(script_path)
    elif script_text:
        text = script_text.strip()
        sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
    else:
        raise ValueError("Either script_path or script_text must be provided")

    custom_badwords = load_custom_badwords(custom_badwords_path) if custom_badwords_path else set()

    tagger = Okt()
    kkma = Kkma()

    stats = analyze_script(sentences, tagger, kkma, custom_badwords)
    length_stats = evaluate_length(text, speech_minutes)

    actual_chars, min_chars, max_chars, length_feedback = length_stats
    return {
        "length": actual_chars,
        "min_length": min_chars,
        "max_length": max_chars,
        "length_feedback": length_feedback,
        **stats,
    }
