

import re
from typing import List, Dict, Tuple, Optional, Set
from src.models import FactualClaim, RelevantChunk, SourceCitation

# Negation PATTERNS — only these trigger the negation guard.
# Uses regex patterns that match genuine grammatical negation, avoiding false-triggers
# on neutral words like "without" or "nor".
_NEGATION_PATTERNS = [
    r"\b(?:was|is|were|did|does|has|had|will|would|can|could|should)\s+not\b",
    r"\b(?:wasn't|isn't|weren't|didn't|doesn't|hasn't|hadn't|won't|wouldn't|can't|couldn't|shouldn't)\b",
    r"\bnever\s+\w+",
    r"\bfailed\s+to\b",
    r"\brefused\s+to\b",
    r"\bdenied\s+\w+",
    r"\bno\s+(?:longer|more|such|evidence|proof|record)\b",
]
_NEGATION_RE = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE)

# Month alias mapping to ensure 'Feb' matches 'February'
_MONTH_STEMS = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr",
    "may": "may", "june": "jun", "july": "jul", "august": "aug",
    "september": "sep", "october": "oct", "november": "nov", "december": "dec",
    "jan": "jan", "feb": "feb", "mar": "mar", "apr": "apr",
    "jun": "jun", "jul": "jul", "aug": "aug", "sep": "sep", "sept": "sep",
    "oct": "oct", "nov": "nov", "dec": "dec",
}


def _split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences while protecting abbreviations, initials, and dates from being severed."""
    if not text:
        return []
    # Protected tokens
    abbrevs = [
        "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
        "U.S.", "U.K.", "D.C.", "e.g.", "i.e.", "etc.", "vs.", "al.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.",
        "No.", "Vol.", "Fig.", "St.", "JPL.", "NASA.", "EDLS.", "Gov.", "Sen.", "Rep."
    ]
    protected = text
    for ab in abbrevs:
        safe = ab.replace(".", "___DOT___")
        protected = re.sub(re.escape(ab), safe, protected, flags=re.IGNORECASE)
    # Protect single uppercase middle initials like 'E.' in 'Octavia E. Butler'
    protected = re.sub(r"\b([A-Z])\.", r"\1___DOT___", protected)
    # Protect decimal numbers like 20.4 or 3.15
    protected = re.sub(r"(\d)\.(\d)", r"\1___DOT___\2", protected)
    # Split on sentence-ending punctuation followed by whitespace
    raw_sents = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [s.replace("___DOT___", ".").strip() for s in raw_sents if s.strip()]
    return sentences


def _clean_claim_text(raw_text: str) -> str:
    """Strips citation markers, markdown bold prefixes, and formatting labels from claim text."""
    # 1. Remove leading markdown bullets (*, -, •, 1.)
    cleaned = re.sub(r"^(?:[\*\-•]\s*|\d+[\.\)]\s*)+", "", raw_text).strip()
    # 2. Remove leading label prefixes like '**Landing Year and Date:**' or '**Official NASA Confirmation:**'
    cleaned = re.sub(r"^(?:\*\*[^*]+:\*\*|[A-Za-z0-9\s/\&,-]+:\*{0,2})\s*", "", cleaned).strip()
    # 3. Preserve 'Source [1]' as 'Source 1' instead of stripping to dangling 'Source '
    cleaned = re.sub(r"\bSource\s*\[(\d+(?:,\s*\d+)*)\]", r"Source \1", cleaned)
    # 4. Remove remaining bracketed citations like [1], [2, 3]
    cleaned = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", cleaned).strip()
    # 5. Clean up any trailing/dangling formatting artifacts
    cleaned = re.sub(r"^\*+\s*", "", cleaned).strip()
    cleaned = re.sub(r"\bSource\s*,", "Source,", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


class ClaimVerifier:
    """Verifies factual claims against retrieved evidence with strict sentence-level entailment."""

    @staticmethod
    def _split_compound_claim(sentence: str) -> List[str]:
        """
        Splits compound sentences that bundle multiple sources or dates into
        individual atomic claims, e.g. 'NASA news release was published on Feb 18, 2021 [1], while Source 2 was published in 2017 [2]'.
        """
        cites = re.findall(r"\[\d+(?:,\s*\d+)*\]", sentence)
        if len(cites) <= 1:
            return [sentence]
        # Split on semicolons or comparative conjunctions preceding another citation
        pattern = r";\s*|,\s*(?:while|whereas|meanwhile|in contrast|and a separate|and another|additionally)\s*"
        parts = re.split(pattern, sentence, flags=re.IGNORECASE)
        clauses = []
        for p in parts:
            p_strip = p.strip()
            if re.search(r"\[\d+\]", p_strip) and len(p_strip) > 20:
                clauses.append(p_strip)
        return clauses if len(clauses) > 1 else [sentence]

    def extract_claims(self, text: str, key_findings: List[str]) -> List[FactualClaim]:
        """Extracts candidate factual claims with citations from key findings and synthesis text.
        
        Compound sentences bundling multiple source citations and dates are automatically
        split into atomic, source-specific claims.
        """
        claims: List[FactualClaim] = []
        # 1. Process explicit key findings
        for kf in key_findings:
            clauses = self._split_compound_claim(kf)
            for clause in clauses:
                cited_ids = self._extract_citation_ids(clause)
                claim_text = _clean_claim_text(clause)
                if len(claim_text) > 15 and not any(c.claim_text.lower() == claim_text.lower() for c in claims):
                    claims.append(
                        FactualClaim(
                            claim_text=claim_text,
                            cited_source_ids=cited_ids,
                            verification_status="unverified",
                            supporting_excerpt="",
                            confidence_score=0.0,
                            reasoning="",
                            entailment_type="none",
                        )
                    )
        # 2. Extract cited sentences from synthesis text
        sentences = _split_into_sentences(text)
        for s in sentences:
            s_clean = s.strip()
            if "[" in s_clean and "]" in s_clean and len(s_clean) > 25:
                clauses = self._split_compound_claim(s_clean)
                for clause in clauses:
                    cited_ids = self._extract_citation_ids(clause)
                    if cited_ids:
                        claim_clean = _clean_claim_text(clause)
                        if len(claim_clean) > 20 and not any(c.claim_text.lower() == claim_clean.lower() for c in claims):
                            claims.append(
                                FactualClaim(
                                    claim_text=claim_clean,
                                    cited_source_ids=cited_ids,
                                    verification_status="unverified",
                                    supporting_excerpt="",
                                    confidence_score=0.0,
                                    reasoning="",
                                    entailment_type="none",
                                )
                            )
        return claims[:8]  # Audit top 8 claims

    def verify_claims(
        self,
        claims: List[FactualClaim],
        chunks: List[RelevantChunk],
        citations: List[SourceCitation],
        groq_client=None,
    ) -> Tuple[List[FactualClaim], float]:
        """
        Validates citation existence, checks DIRECT ENTAILMENT of key entities/dates/numbers
        at the SENTENCE level, verifies webpage metadata separately from article text,
        and computes a transparent grounding score.

        Grounding Score Formula (documented):
            score = (verified * 1.0 + qualified * 0.5 + unsupported * 0.0) / total * 100
        """
        if not claims:
            return [], 0.0
        # Deduplicate claims to avoid inflating grounding score
        seen = set()
        unique_claims = []
        for cl in claims:
            norm = cl.claim_text.strip().lower()
            if norm not in seen:
                seen.add(norm)
                unique_claims.append(cl)
        claims = unique_claims
        valid_source_ids = {c.id for c in citations}
        source_texts_by_id: Dict[int, List[str]] = {}
        for c in chunks:
            source_texts_by_id.setdefault(c.source_index, []).append(c.text)
        # Inject citation metadata so claims about publication dates, event dates, and domains can verify
        for cit in citations:
            meta_entries = []
            if cit.published_date:
                meta_entries.append(f"{cit.title} was published on {cit.published_date} by {cit.domain}.")
                try:
                    from datetime import datetime
                    dt = datetime.strptime(cit.published_date, "%Y-%m-%d")
                    meta_entries.append(f"{cit.title} has a publication date of {dt.strftime('%B %d, %Y')} on {cit.domain}.")
                except Exception:
                    pass
            if cit.event_date:
                meta_entries.append(f"Event date documented by {cit.domain}: {cit.event_date}.")
            if cit.snippet:
                meta_entries.append(cit.snippet)
            if cit.title:
                meta_entries.append(cit.title)
            source_texts_by_id.setdefault(cit.id, []).extend(meta_entries)
        weighted_score_sum = 0.0
        for claim in claims:
            # --- Step 1: Validate citation IDs exist ---
            if not claim.cited_source_ids:
                claim.verification_status = "unsupported"
                claim.confidence_score = 0.0
                claim.reasoning = "No source citation attached — this statement is unverifiable as written"
                claim.entailment_type = "none"
                continue
            invalid_ids = [cid for cid in claim.cited_source_ids if cid not in valid_source_ids]
            if invalid_ids:
                claim.verification_status = "unsupported"
                claim.confidence_score = 0.0
                claim.reasoning = f"Cited source ID(s) {invalid_ids} do not exist in the source registry"
                claim.entailment_type = "none"
                continue
            # --- Dedicated Webpage Metadata Verification ---
            is_pub_date_claim = bool(
                re.search(
                    r"\b(?:published|publication\\s+date|metadata\\s+date|webpage\\s+date|posted\\s+on)\b",
                    claim.claim_text,
                    re.IGNORECASE,
                )
            )
            if is_pub_date_claim:
                meta_match = None
                for cid in claim.cited_source_ids:
                    cit = next((c for c in citations if c.id == cid), None)
                    if cit and cit.published_date:
                        pyear = cit.published_date[:4]
                        if pyear in claim.claim_text:
                            meta_match = cit
                            break
                if meta_match:
                    claim.verification_status = "verified"
                    claim.confidence_score = 0.98
                    claim.reasoning = (
                        f"Webpage metadata verified: publication date {meta_match.published_date} "
                        f"confirmed for Source [{meta_match.id}] ({meta_match.domain})"
                    )
                    claim.supporting_excerpt = (
                        f"{meta_match.title} — Official publication date in metadata: "
                        f"{meta_match.published_date} ({meta_match.domain})"
                    )
                    claim.entailment_type = "direct"
                    weighted_score_sum += 1.0
                    continue
            # --- Step 2: Collect text from cited sources ---
            relevant_texts: List[str] = []
            for cid in claim.cited_source_ids:
                relevant_texts.extend(source_texts_by_id.get(cid, []))
            combined_source = " ".join(relevant_texts)
            if not combined_source.strip():
                claim.verification_status = "unsupported"
                claim.confidence_score = 0.0
                claim.reasoning = "Cited source contains no readable text after scraping"
                claim.entailment_type = "none"
                continue
            # --- Step 3: Detect genuine negation in claim ---
            claim_has_negation = bool(_NEGATION_RE.search(claim.claim_text))
            # --- Step 4: Extract key entities, dates, numbers from the claim ---
            claim_dates = re.findall(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
                r"\s+\d{1,2},?\s+\d{4}\b"
                r"|\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
                r"\s+\d{4}\b"
                r"|\b\d{4}-\d{2}-\d{2}\b"
                r"|\b(?:19|20)\d{2}\b",
                claim.claim_text,
                re.IGNORECASE,
            )
            # Comma-aware numbers: e.g. 2,000 or 148.8 or 515.31
            claim_numbers = re.findall(
                r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?|\b\d+(?:\.\d+)?(?:\s*(?:crore|lakh|million|billion|trillion|%|percent|MW|GW|MJ|qubits|seconds|minutes|hours|days|years|sols))?",
                claim.claim_text,
                re.IGNORECASE,
            )
            # Proper nouns / named entities (including acronyms like NASA, JPL, UTC)
            raw_entities = re.findall(r"\b(?:[A-Z]{2,}|[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b", claim.claim_text)
            stop_entities = {"The", "This", "These", "Those", "There", "When", "What", "Where", "How", "According", "For"}
            claim_entities = [e for e in raw_entities if e not in stop_entities]
            # --- Step 5: Find the single best matching sentence in the source ---
            best_sentence, best_overlap = self._find_best_entailing_sentence(claim.claim_text, relevant_texts)
            claim.supporting_excerpt = best_sentence[:300] if best_sentence else ""
            best_sent_lower = best_sentence.lower() if best_sentence else ""
            source_lower = combined_source.lower()
            # --- Step 6: Validate dates and numbers ---
            date_mismatches_in_sentence = []
            for d in claim_dates:
                year_match = re.search(r"(?:19|20)\d{2}", d)
                year_str = year_match.group() if year_match else ""
                # Also check month if present
                month_match = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", d, re.IGNORECASE)
                month_stem = _MONTH_STEMS.get(month_match.group(1).lower()) if month_match else ""
                if len(d) == 4:
                    if d not in best_sent_lower:
                        date_mismatches_in_sentence.append(d)
                else:
                    has_year = year_str and (year_str in best_sent_lower)
                    has_month = (not month_stem) or (month_stem in best_sent_lower)
                    if not (has_year and has_month):
                        date_mismatches_in_sentence.append(d)
            numbers_missing_from_source = []
            numbers_in_source_not_sentence = []
            for num in claim_numbers:
                core = re.sub(r"[^\d.]", "", num)
                # Ignore single digits (0-9) as too generic
                if len(core) >= 3:
                    if core not in best_sent_lower:
                        # Check normalized version (e.g. 2000 vs 2,000)
                        if core in source_lower or num.lower() in source_lower:
                            numbers_in_source_not_sentence.append(num)
                        else:
                            numbers_missing_from_source.append(num)
            # --- Step 7: Entity presence ratio in best sentence ---
            entity_hits = 0
            for e in claim_entities:
                e_lower = e.lower()
                if e_lower in best_sent_lower or _MONTH_STEMS.get(e_lower, "###") in best_sent_lower:
                    entity_hits += 1
            entity_ratio = entity_hits / len(claim_entities) if claim_entities else 1.0
            # --- Step 8: Broad topic overlap ---
            claim_word_set = set(re.findall(r"\b\w{4,}\b", claim.claim_text.lower()))
            source_word_set = set(re.findall(r"\b\w{4,}\b", combined_source.lower()))
            overlap_ratio = len(claim_word_set & source_word_set) / len(claim_word_set) if claim_word_set else 0.0
            is_projection = bool(
                re.search(
                    r"\b(?:future target|long-term goal|projected to|roadmap|aims to|expected to|by 203\d|by 204\d)\b",
                    claim.claim_text,
                    re.IGNORECASE,
                )
            )
            # --- Step 9: Determine verification status ---
            if claim_has_negation:
                if best_overlap >= 4 and entity_ratio >= 0.5:
                    claim.verification_status = "qualified"
                    claim.confidence_score = 0.60
                    claim.reasoning = (
                        "Claim contains negation — source excerpt discusses the same topic but "
                        "lexical overlap cannot confirm or refute the negated assertion"
                    )
                    claim.entailment_type = "indirect"
                else:
                    claim.verification_status = "unsupported"
                    claim.confidence_score = 0.20
                    claim.reasoning = (
                        "Claim contains negation and the source excerpt does not confirm it — "
                        "insufficient overlap to verify the negated statement"
                    )
                    claim.entailment_type = "none"
            elif numbers_missing_from_source:
                claim.verification_status = "unsupported"
                claim.confidence_score = 0.20
                claim.reasoning = (
                    f"Key figure(s) {numbers_missing_from_source} from the claim are not found "
                    f"anywhere in the cited source text"
                )
                claim.entailment_type = "none"
            elif numbers_in_source_not_sentence:
                if len(claim.cited_source_ids) > 1:
                    claim.verification_status = "verified"
                    claim.confidence_score = 0.85
                    claim.reasoning = (
                        f"Multi-source synthesis: figures {numbers_in_source_not_sentence} verified "
                        f"across cited sources {claim.cited_source_ids}"
                    )
                    claim.entailment_type = "direct"
                else:
                    claim.verification_status = "qualified"
                    claim.confidence_score = 0.65
                    claim.reasoning = (
                        f"Compound claim: figure(s) {numbers_in_source_not_sentence} exist in the "
                        f"cited source but are spread across multiple sentences — partial sentence entailment"
                    )
                    claim.entailment_type = "indirect"
            elif date_mismatches_in_sentence and entity_ratio < 0.5:
                claim.verification_status = "unsupported"
                claim.confidence_score = 0.30
                claim.reasoning = (
                    f"Date(s) {date_mismatches_in_sentence} and key entities not confirmed "
                    f"in best-matching source sentence"
                )
                claim.entailment_type = "none"
            elif best_overlap >= 3 and entity_ratio >= 0.5:
                # Entailment matches: sentence contains key facts and entities
                if is_projection:
                    claim.verification_status = "qualified"
                    claim.confidence_score = 0.80
                    claim.reasoning = "Forward-looking target or roadmap; context confirmed in source sentence"
                    claim.entailment_type = "indirect"
                else:
                    claim.verification_status = "verified"
                    claim.confidence_score = round(min(0.98, 0.76 + best_overlap * 0.02), 2)
                    claim.reasoning = "Key entities, dates, and context directly matched in cited source sentence"
                    claim.entailment_type = "direct"
            elif overlap_ratio >= 0.40:
                claim.verification_status = "qualified"
                claim.confidence_score = 0.60
                claim.reasoning = (
                    "Claim is topically aligned with source but paraphrased — "
                    "no direct sentence entailment found"
                )
                claim.entailment_type = "indirect"
            else:
                claim.verification_status = "unsupported"
                claim.confidence_score = 0.20
                claim.reasoning = "Insufficient lexical and factual entailment with cited passage"
                claim.entailment_type = "none"
            # --- Step 10: Accumulate weighted score ---
            if claim.verification_status == "verified":
                weighted_score_sum += 1.0
            elif claim.verification_status == "qualified":
                weighted_score_sum += 0.5
        grounding_score = round((weighted_score_sum / len(claims)) * 100.0, 1)
        return claims, grounding_score

    def _find_best_entailing_sentence(self, claim_text: str, source_texts: List[str]) -> Tuple[str, int]:
        """
        Finds the single sentence (or consecutive sentence pair) in source_texts with the highest
        entailment overlap with the claim, weighting numbers, acronyms, and lexical stems.
        """
        claim_words = set(re.findall(r"\b[A-Za-z0-9_]{3,}\b", claim_text.lower()))
        for word in re.findall(r"\b[A-Za-z]{3}\b", claim_text.lower()):
            if word in _MONTH_STEMS:
                claim_words.add(_MONTH_STEMS[word])
        claim_nums = set(re.findall(r"\b\d+\b", claim_text))
        claim_acronyms = set(re.findall(r"\b[A-Z]{2,}\b", claim_text))
        if not claim_words and not claim_nums:
            return (source_texts[0][:200] if source_texts else ""), 0
        # Build candidate passages (both individual sentences and adjacent pairs)
        candidate_passages = []
        for text in source_texts:
            sentences = _split_into_sentences(text)
            for i, s in enumerate(sentences):
                s_clean = s.strip()
                if len(s_clean) >= 20:
                    candidate_passages.append(s_clean)
                if i + 1 < len(sentences):
                    pair = (s_clean + " " + sentences[i+1].strip()).strip()
                    if len(pair) >= 30 and len(pair) <= 450:
                        candidate_passages.append(pair)
        best_sentence = ""
        best_score = 0
        for s in candidate_passages:
            s_lower = s.lower()
            s_words = set(re.findall(r"\b[A-Za-z0-9_]{3,}\b", s_lower))
            for word in re.findall(r"\b[A-Za-z]{3}\b", s_lower):
                if word in _MONTH_STEMS:
                    s_words.add(_MONTH_STEMS[word])
            s_nums = set(re.findall(r"\b\d+\b", s))
            s_acronyms = set(re.findall(r"\b[A-Z]{2,}\b", s))
            # Base lexical word overlap
            score = len(claim_words & s_words)
            # High weight on matching exact numbers (e.g. 20, 55, 18, 2021)
            score += 3 * len(claim_nums & s_nums)
            # High weight on matching acronyms (e.g. UTC, JPL, NASA)
            score += 3 * len(claim_acronyms & s_acronyms)
            # Stem prefix boost for common words
            for cw in claim_words:
                if len(cw) >= 5:
                    stem = cw[:5]
                    if any(sw.startswith(stem) for sw in s_words):
                        score += 1
            if score > best_score:
                best_score = score
                best_sentence = s.strip()
        return best_sentence, best_score

    @staticmethod
    def _extract_citation_ids(text: str) -> List[int]:
        """Extracts integers from citation markers like [1], [1, 2], [3]."""
        matches = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text)
        ids = []
        for m in matches:
            for num in re.split(r",\s*", m):
                if num.isdigit():
                    ids.append(int(num))
        return sorted(list(set(ids)))
