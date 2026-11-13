#!/usr/bin/env python3
"""Build sentence-level disaster NER scope records from document-level JSONL data.

This module converts document-level Vietnamese disaster articles into sentence-level
JSONL records for downstream annotation scoping. The implementation is entirely
rule-based, deterministic, and optimized for high recall so that potentially
useful disaster-related sentences are retained.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


logger = logging.getLogger("build_scoped_sentences")


@dataclass
class SentenceRecord:
    """Structured sentence-level record for output."""

    sentence_id: str
    document_id: str
    sentence_index: int
    text: str
    keep_for_annotation: bool
    matched_categories: Dict[str, List[str]]
    scope_reasons: List[str]
    source_name: str
    source_type: str
    url: str
    title: str
    publication_date: str
    document_keyword_score: int
    document_matched_keywords: List[str]
    document_quality_flags: List[str]
    char_start: int
    char_end: int
    text_length: int
    language: str

    def to_payload(self) -> Dict[str, Any]:
        """Convert to the exact output schema expected by the dataset."""
        payload = asdict(self)
        payload["matched_categories"] = {
            key: payload["matched_categories"].get(key, []) for key in [
                "disaster_type",
                "weather_cause",
                "impact",
                "casualty",
                "affected_population",
                "response",
                "agency",
                "warning",
            ]
        }
        return payload


class SentenceScopeBuilder:
    """Rule-based sentence splitter and scope classifier for disaster content."""

    CATEGORY_NAMES = [
        "disaster_type",
        "weather_cause",
        "impact",
        "casualty",
        "affected_population",
        "response",
        "agency",
        "warning",
    ]

    def __init__(self) -> None:
        self.lexicon = self._build_lexicon()
        self.abbreviations = {
            "bs",
            "dr",
            "gs",
            "mr",
            "mrs",
            "pgs",
            "p",
            "q",
            "th.s",
            "ths",
            "tp",
            "ts",
        }
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        self.boilerplate_patterns = [
            r"^\s*(?:ảnh minh họa|chú thích ảnh|hình ảnh|ảnh|photo|caption|image)\s*[:：–—-]",
            r"^\s*(?:copyright|bản quyền)\b",
            r"^\s*(?:advertisement|quảng cáo|quảng bá)\b",
            r"^\s*(?:share|chia sẻ|like|comment)\b",
            r"^\s*(?:navigation|điều hướng|menu)\b",
            r"^\s*(?:related articles|bài viết liên quan|tin liên quan|tin mới nhất|tin cùng chuyên mục|xem nhiều|xem thêm|see also|đọc thêm)\b",
            r"^\s*(?:video|video label|nhãn video)\s*[:：–—-]",
            r"^\s*(?:read more|see also)\b",
        ]
        self.boilerplate_compiled = [re.compile(pattern, re.IGNORECASE) for pattern in self.boilerplate_patterns]
        self.compiled_lexicon = {
            category: [
                (
                    phrase,
                    re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE),
                )
                for phrase, _ in sorted(
                    phrases,
                    key=lambda item: len(item[0]),
                    reverse=True,
                )
            ]
            for category, phrases in self.lexicon.items()
        }
        self.core_impact_terms = {
            "ngập",
            "ngập sâu",
            "ngập úng",
            "sạt lở",
            "thiệt hại",
            "hư hỏng",
            "mất điện",
            "tê liệt",
            "chia cắt",
            "cô lập",
            "mắc kẹt",
            "cuốn trôi",
            "gãy đổ",
            "hủy hoại",
            "xâm nhập nước",
        }
        self.evacuation_terms = {"phải sơ tán", "sơ tán", "di dời", "di tản"}
        self.contextual_response_terms = {
            "cứu hộ",
            "cứu nạn",
            "ứng phó",
            "phòng chống",
            "sơ tán",
            "di tản",
            "tìm kiếm",
            "cứu trợ",
        }
        self.direct_warning_terms = {"cấm biển", "cấm tàu thuyền"}

    def _build_lexicon(self) -> Dict[str, List[Tuple[str, str]]]:
        """Create a lexicon of Vietnamese disaster-related phrases."""
        return {
            "disaster_type": [
                ("bão", "bão"),
                ("lũ", "lũ"),
                ("lụt", "lụt"),
                ("lũ quét", "lũ quét"),
                ("sạt lở", "sạt lở"),
                ("sạt lở đất", "sạt lở đất"),
                ("ngập lụt", "ngập lụt"),
                ("mưa lớn", "mưa lớn"),
                ("mưa to", "mưa to"),
                ("mưa", "mưa"),
                ("mưa cực đoan", "mưa cực đoan"),
                ("hạn hán", "hạn hán"),
                ("cháy rừng", "cháy rừng"),
                ("hỏa hoạn", "hỏa hoạn"),
                ("động đất", "động đất"),
                ("tuyết rơi", "tuyết rơi"),
                ("nắng nóng", "nắng nóng"),
                ("không khí lạnh", "không khí lạnh"),
                ("lốc xoáy", "lốc xoáy"),
                ("mưa đá", "mưa đá"),
                ("nước dâng", "nước dâng"),
                ("nước lũ", "nước lũ"),
                ("hồ chứa vỡ", "hồ chứa vỡ"),
                ("cát lũ", "cát lũ"),
                ("trượt lở", "trượt lở"),
                ("lũ quét", "lũ quét"),
            ],
            "weather_cause": [
                ("mưa lớn", "mưa lớn"),
                ("mưa to", "mưa to"),
                ("mưa", "mưa"),
                ("mưa liên tục", "mưa liên tục"),
                ("mưa kéo dài", "mưa kéo dài"),
                ("nước dâng", "nước dâng"),
                ("gió mạnh", "gió mạnh"),
                ("gió lớn", "gió lớn"),
                ("bão", "bão"),
                ("hoàn lưu bão", "hoàn lưu bão"),
                ("xả lũ", "xả lũ"),
                ("nắng nóng", "nắng nóng"),
                ("không khí lạnh", "không khí lạnh"),
                ("sương mù", "sương mù"),
                ("lốc", "lốc"),
                ("giông", "giông"),
            ],
            "impact": [
                ("ngập", "ngập"),
                ("ngập sâu", "ngập sâu"),
                ("ngập úng", "ngập úng"),
                ("sạt lở", "sạt lở"),
                ("thiệt hại", "thiệt hại"),
                ("hư hỏng", "hư hỏng"),
                ("mất điện", "mất điện"),
                ("tê liệt", "tê liệt"),
                ("chia cắt", "chia cắt"),
                ("cô lập", "cô lập"),
                ("nguy hiểm", "nguy hiểm"),
                ("mắc kẹt", "mắc kẹt"),
                ("cuốn trôi", "cuốn trôi"),
                ("gãy đổ", "gãy đổ"),
                ("đổ", "đổ"),
                ("trôi", "trôi"),
                ("hủy hoại", "hủy hoại"),
                ("xâm nhập nước", "xâm nhập nước"),
            ],
            "casualty": [
                ("chết", "chết"),
                ("tử vong", "tử vong"),
                ("mất tích", "mất tích"),
                ("bị thương", "bị thương"),
                ("tử thương", "tử thương"),
                ("chết người", "chết người"),
                ("thiệt mạng", "thiệt mạng"),
                ("người chết", "người chết"),
                ("người mất tích", "người mất tích"),
                ("nạn nhân", "nạn nhân"),
            ],
            "affected_population": [
                ("hộ dân", "hộ dân"),
                ("người dân", "người dân"),
                ("hộ gia đình", "hộ gia đình"),
                ("phải sơ tán", "phải sơ tán"),
                ("sơ tán", "sơ tán"),
                ("di dời", "di dời"),
                ("di tản", "di tản"),
                ("bị ảnh hưởng", "bị ảnh hưởng"),
                ("khu dân cư", "khu dân cư"),
                ("cư dân", "cư dân"),
            ],
            "response": [
                ("cứu hộ", "cứu hộ"),
                ("cứu nạn", "cứu nạn"),
                ("ứng phó", "ứng phó"),
                ("phòng chống", "phòng chống"),
                ("xử lý", "xử lý"),
                ("khắc phục", "khắc phục"),
                ("hỗ trợ", "hỗ trợ"),
                ("tìm kiếm", "tìm kiếm"),
                ("tiếp cận", "tiếp cận"),
                ("phương án ứng cứu", "phương án ứng cứu"),
                ("sơ tán", "sơ tán"),
                ("di tản", "di tản"),
                ("cấp cứu", "cấp cứu"),
                ("điều phối", "điều phối"),
                ("cứu trợ", "cứu trợ"),
            ],
            "agency": [
                ("công an", "công an"),
                ("quân đội", "quân đội"),
                ("dân quân", "dân quân"),
                ("chính quyền", "chính quyền"),
                ("ubnd", "ubnd"),
                ("ban chỉ huy", "ban chỉ huy"),
                ("cục cứu hộ", "cục cứu hộ"),
                ("cảnh sát", "cảnh sát"),
                ("lực lượng chức năng", "lực lượng chức năng"),
                ("đài khí tượng", "đài khí tượng"),
                ("cục khí tượng", "cục khí tượng"),
                ("cơ quan khí tượng", "cơ quan khí tượng"),
                ("hải quân", "hải quân"),
                ("đặc công", "đặc công"),
            ],
            "warning": [
                ("cảnh báo", "cảnh báo"),
                ("đề phòng", "đề phòng"),
                ("nguy cơ", "nguy cơ"),
                ("rủi ro", "rủi ro"),
                ("cảnh giác", "cảnh giác"),
                ("khẩn cấp", "khẩn cấp"),
                ("nguy hiểm", "nguy hiểm"),
                ("cấm biển", "cấm biển"),
                ("cấm tàu thuyền", "cấm tàu thuyền"),
            ],
        }

    def normalize_whitespace(self, text: str) -> str:
        """Collapse repeated whitespace while preserving visible content."""
        return re.sub(r"\s+", " ", text).strip()

    def _iter_sentences(self, text: str) -> List[Tuple[int, int]]:
        """Find sentence spans while retaining offsets into the original text."""
        spans: List[Tuple[int, int]] = []
        if not text:
            return spans

        i = 0
        length = len(text)
        while i < length:
            while i < length and text[i].isspace():
                i += 1
            if i >= length:
                break
            start = i
            while i < length:
                char = text[i]
                if char in ".!?…":
                    if self._is_sentence_boundary(text, i):
                        end = self._advance_past_closure(text, i)
                        while end > start and text[end - 1].isspace():
                            end -= 1
                        if end > start:
                            spans.append((start, end))
                        i = self._advance_past_closure(text, i)
                        break
                elif char in "\r\n":
                    end = i
                    while end > start and text[end - 1].isspace():
                        end -= 1
                    if end > start:
                        spans.append((start, end))
                    while i < length and text[i].isspace():
                        i += 1
                    break
                i += 1
            else:
                spans.append((start, length))
                break
            if i < length and text[i].isspace():
                while i < length and text[i].isspace():
                    i += 1
        return spans

    def _advance_past_closure(self, text: str, index: int) -> int:
        """Extend the sentence span past closing quotes or parentheses after punctuation."""
        end = index + 1
        while end < len(text) and text[end] in ".!?…\"'’”»)]}":
            end += 1
        return end

    def _is_sentence_boundary(self, text: str, index: int) -> bool:
        """Return True when a punctuation mark likely closes a sentence."""
        if not text:
            return False
        char = text[index]
        if char in ".!?":
            if char == ".":
                if index + 1 < len(text) and text[index + 1] == ".":
                    return False
                if self._looks_like_decimal(text, index):
                    return False
                if self._is_url_or_email_fragment(text, index):
                    return False
                if self._is_abbreviation(text, index):
                    return False
        return char in ".!?…"

    def _looks_like_decimal(self, text: str, index: int) -> bool:
        """Avoid splitting decimals such as 3.5 or 1.5m."""
        return (
            index > 0
            and index + 1 < len(text)
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        )

    def _is_url_or_email_fragment(self, text: str, index: int) -> bool:
        """Avoid splitting on URL or email punctuation."""
        if index <= 0:
            return False
        start = index
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        end = index + 1
        while end < len(text) and not text[end].isspace():
            end += 1
        token = text[start:end]
        return bool(
            re.search(r"https?://|www\.|[\w.+-]+@|[\w.-]+\.(?:com|vn|org|net|gov|edu)\b", token, re.IGNORECASE)
        )

    def _is_abbreviation(self, text: str, index: int) -> bool:
        """Ignore sentence splits after common Vietnamese abbreviations."""
        if index <= 0:
            return False
        start = index
        while start > 0 and text[start - 1].isalpha():
            start -= 1
        token = text[start:index]
        if not token:
            return False
        if token.lower() not in self.abbreviations:
            return False
        next_idx = self._next_non_space_index(text, index + 1)
        return next_idx is not None and text[next_idx].isupper()

    def _next_non_space_index(self, text: str, start: int) -> Optional[int]:
        """Find next non-whitespace character index."""
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            return None
        return start

    def split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split a document into normalized sentences with original offsets."""
        spans = self._iter_sentences(text)
        sentences: List[Tuple[str, int, int]] = []
        for start, end in spans:
            if end <= start:
                continue
            raw = text[start:end]
            normalized = self.normalize_whitespace(raw)
            if not normalized:
                continue
            sentences.append((normalized, start, end))
        return sentences

    def _match_keyword(self, text: str, keyword: str) -> Optional[str]:
        """Find a whole-phrase match and return its original surface form."""
        if not keyword:
            return None
        pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(0)
        return None

    def match_categories(self, text: str) -> Dict[str, List[str]]:
        """Match disaster-related lexicon terms and surface forms per category."""
        matches: Dict[str, List[str]] = {name: [] for name in self.CATEGORY_NAMES}
        for category in self.CATEGORY_NAMES:
            occupied_spans: List[Tuple[int, int]] = []
            for phrase, pattern in self.compiled_lexicon.get(category, []):
                for match in pattern.finditer(text):
                    if phrase == "bão" and re.match(
                        r"\s+(?:hoà|hòa)\b",
                        text[match.end():],
                        re.IGNORECASE,
                    ):
                        continue
                    span = match.span()
                    if any(span[0] < end and start < span[1] for start, end in occupied_spans):
                        continue
                    occupied_spans.append(span)
                    original = match.group(0)
                    if original not in matches[category]:
                        matches[category].append(original)
        return matches

    def _contains_boilerplate(self, text: str) -> bool:
        """Reject obvious boilerplate or non-event content."""
        lowered = text.lower()
        stripped = lowered.strip()
        if not stripped:
            return True
        if self.url_pattern.fullmatch(stripped):
            return True
        for pattern in self.boilerplate_compiled:
            if pattern.search(stripped):
                return True
        if stripped.startswith("đang tải") or stripped.startswith("đăng nhập"):
            return True
        if stripped.startswith("chú giải"):
            return True
        if "thêm chuyên mục" in stripped and "tuổi trẻ" in stripped:
            return True
        return False

    def evaluate_scope(self, text: str, matched_categories: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
        """Rule-based scope decision with explainable reasons."""
        if self._contains_boilerplate(text):
            return False, []

        reasons: List[str] = []
        disaster_terms = matched_categories["disaster_type"]
        weather_terms = matched_categories["weather_cause"]
        impact_terms = matched_categories["impact"]
        casualty_terms = matched_categories["casualty"]
        population_terms = matched_categories["affected_population"]
        response_terms = matched_categories["response"]
        agency_terms = matched_categories["agency"]

        if disaster_terms:
            reasons.append("has_disaster_type")
        if weather_terms:
            reasons.append("has_weather_cause")
        if impact_terms or casualty_terms:
            reasons.append("has_damage")
        if casualty_terms:
            reasons.append("has_casualty")
        if population_terms:
            reasons.append("has_affected_population")
        if response_terms:
            reasons.append("has_response")
        warning_terms = matched_categories["warning"]
        if warning_terms and (
            disaster_terms
            or weather_terms
            or self.direct_warning_terms.intersection(term.lower() for term in warning_terms)
        ):
            reasons.append("has_warning")

        non_empty_categories = sum(1 for category in self.CATEGORY_NAMES if matched_categories[category])
        if non_empty_categories >= 2:
            reasons.append("multi_category")

        has_direct_event_signal = bool(
            disaster_terms
            or weather_terms
            or casualty_terms
            or self.core_impact_terms.intersection(term.lower() for term in impact_terms)
            or self.evacuation_terms.intersection(term.lower() for term in population_terms)
            or self.direct_warning_terms.intersection(term.lower() for term in warning_terms)
        )
        has_response_context = bool(
            self.contextual_response_terms.intersection(term.lower() for term in response_terms)
            and (agency_terms or population_terms or casualty_terms)
        )
        if has_direct_event_signal or has_response_context:
            reasons.append("contains_event_information")

        should_keep = has_direct_event_signal or has_response_context
        if should_keep:
            reasons.append("sentence_has_event_info")
        return should_keep, reasons

    def process_document(self, document: Dict[str, Any], sentence_index_offset: int = 0) -> List[SentenceRecord]:
        """Create scoped sentence records for a single document."""
        text = str(document.get("clean_text") or document.get("text") or "")
        sentences = self.split_sentences(text)
        records: List[SentenceRecord] = []
        for local_index, (sentence_text, start, end) in enumerate(sentences):
            matched_categories = self.match_categories(sentence_text)
            keep_for_annotation, reasons = self.evaluate_scope(sentence_text, matched_categories)
            sentence_id = f"{document.get('document_id', 'document')}_sent_{sentence_index_offset + local_index:04d}"
            record = SentenceRecord(
                sentence_id=sentence_id,
                document_id=str(document.get("document_id", "") or ""),
                sentence_index=sentence_index_offset + local_index,
                text=sentence_text,
                keep_for_annotation=keep_for_annotation,
                matched_categories=matched_categories,
                scope_reasons=reasons,
                source_name=str(document.get("source_name", "") or ""),
                source_type=str(document.get("source_type", "") or ""),
                url=str(document.get("url", "") or ""),
                title=str(document.get("title", "") or ""),
                publication_date=str(document.get("publication_date", "") or ""),
                document_keyword_score=int(document.get("keyword_score", 0) or 0),
                document_matched_keywords=list(document.get("matched_keywords", []) or []),
                document_quality_flags=list(document.get("quality_flags", []) or []),
                char_start=start,
                char_end=end,
                text_length=len(sentence_text),
                language=str(document.get("language", "vi") or "vi"),
            )
            records.append(record)
        return records

    def process_documents(self, documents: Sequence[Dict[str, Any]]) -> Tuple[List[SentenceRecord], Dict[str, Any]]:
        """Process a batch of documents and return records plus summary stats."""
        records: List[SentenceRecord] = []
        sentence_counter = Counter()
        kept_counter = 0
        rejected_counter = 0
        for document in tqdm(documents, desc="Processing documents"):
            doc_records = self.process_document(document)
            if not doc_records:
                continue
            records.extend(doc_records)
            kept_counter += sum(1 for record in doc_records if record.keep_for_annotation)
            rejected_counter += sum(1 for record in doc_records if not record.keep_for_annotation)
            for record in doc_records:
                for category, values in record.matched_categories.items():
                    sentence_counter[category] += len(values)

        total_documents = len(documents)
        stats = {
            "documents_processed": total_documents,
            "sentences_generated": len(records),
            "sentences_kept": kept_counter,
            "sentences_rejected": rejected_counter,
            "keep_ratio": round(kept_counter / len(records), 4) if records else 0.0,
            "average_sentences_per_document": round(len(records) / total_documents, 2) if total_documents else 0.0,
            "category_frequencies": dict(sorted(sentence_counter.items())),
        }
        return records, stats

    def write_jsonl(self, records: Sequence[SentenceRecord], output_path: Path) -> int:
        """Write only kept records as JSONL, one record per line."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kept_records = [record for record in records if record.keep_for_annotation]
        with output_path.open("w", encoding="utf-8") as handle:
            for record in kept_records:
                handle.write(json.dumps(record.to_payload(), ensure_ascii=False))
                handle.write("\n")
        return len(kept_records)


def load_documents(input_path: Path) -> List[Dict[str, Any]]:
    """Load document JSONL as Python dictionaries."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    documents: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {input_path}: {exc}") from exc
            documents.append(payload)
    return documents


def resolve_input_path(candidate: Optional[str]) -> Path:
    """Resolve an input path from CLI arguments or common workspace names."""
    if candidate:
        return Path(candidate).expanduser().resolve()
    current_dir = Path.cwd()
    for name in ["processed_documents.jsonl", "disaster_documents_expanded (2).jsonl", "disaster_documents_expanded.jsonl"]:
        path = current_dir / name
        if path.exists():
            return path
    if (current_dir / "processed_documents.jsonl").exists():
        return current_dir / "processed_documents.jsonl"
    return current_dir / "processed_documents.jsonl"


def print_statistics(stats: Dict[str, Any]) -> None:
    """Display a compact summary of processing statistics."""
    print("\nStatistics")
    print("----------")
    print(f"Documents processed: {stats['documents_processed']}")
    print(f"Sentences generated: {stats['sentences_generated']}")
    print(f"Sentences kept: {stats['sentences_kept']}")
    print(f"Sentences rejected: {stats['sentences_rejected']}")
    print(f"Keep ratio: {stats['keep_ratio']:.4f}")
    print(f"Average sentences per document: {stats['average_sentences_per_document']:.2f}")
    print("Category frequencies:")
    for category, count in stats["category_frequencies"].items():
        print(f"  - {category}: {count}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build sentence-level disaster scope records from document JSONL")
    parser.add_argument("input_file", nargs="?", default=None, help="Path to the input document JSONL file")
    parser.add_argument("-o", "--output", default=None, help="Path to the output sentence JSONL file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    input_path = resolve_input_path(args.input_file)
    output_path = Path(args.output).expanduser().resolve() if args.output else (input_path.parent / "scoped_sentences.jsonl")

    logger.info("Loading documents from %s", input_path)
    documents = load_documents(input_path)

    builder = SentenceScopeBuilder()
    records, stats = builder.process_documents(documents)
    written_count = builder.write_jsonl(records, output_path)
    print_statistics(stats)
    print(f"Wrote {written_count} kept records to {output_path}")


if __name__ == "__main__":
    main()
