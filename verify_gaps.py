"""Comprehensive gap verification script — tests all audit findings."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.verifier import ClaimVerifier, _NEGATION_RE
from src.retrieval import InformationRetriever
from src.models import FactualClaim, RelevantChunk, SourceCitation, ResearchReport

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} — {detail}")

v = ClaimVerifier()
r = InformationRetriever()

# ===== GAP 1: Negation Bug =====
print("\n" + "="*60)
print("GAP 1: Claim Verification - Negation Bug")
print("="*60)

chunk = RelevantChunk(source_index=1, title='T', url='http://x.com', domain='x.com',
    text='On Friday 17 November 2023, OpenAI board fired Sam Altman, stating he had not been consistently candid.',
    relevance_score=2.0, credibility_badge='x', credibility_score=60)
cit = SourceCitation(id=1, title='T', url='http://x.com', domain='x.com', snippet='x', credibility_badge='x', credibility_score=60)
claim = FactualClaim(claim_text='Sam Altman was not the CEO of OpenAI on November 17, 2023 [1]',
    cited_source_ids=[1], verification_status='unverified')
results, score = v.verify_claims([claim], [chunk], [cit])
check("Negated claim NOT verified", results[0].verification_status != 'verified',
    f"got {results[0].verification_status}")
check("Negated claim NOT direct entailment", results[0].entailment_type != 'direct',
    f"got {results[0].entailment_type}")

# 'without human intervention' should NOT trigger negation
has_neg = bool(_NEGATION_RE.search('AI agents execute tasks without human intervention'))
check("'without' does NOT trigger negation", not has_neg, f"detected={has_neg}")

# ===== GAP 2: Domain Spoofing =====
print("\n" + "="*60)
print("GAP 2: Domain Spoofing Prevention")
print("="*60)

sc, bd = r.evaluate_source_credibility('evilnature.com')
check("evilnature.com blocked", sc < 90 and 'Academic' not in bd, f"score={sc}, badge={bd}")

sc, bd = r.evaluate_source_credibility('example.gov.attacker.com')
check("example.gov.attacker.com blocked", sc < 90 and 'Government' not in bd, f"score={sc}, badge={bd}")

sc, bd = r.evaluate_source_credibility('pmindia.gov.in')
check("pmindia.gov.in recognized", sc >= 90 and 'Government' in bd, f"score={sc}, badge={bd}")

sc, bd = r.evaluate_source_credibility('beginnersinai.org')
check(".org NOT labeled 'Verified'", 'Verified' not in bd, f"badge={bd}")

sc, bd = r.evaluate_source_credibility('en.wikipedia.org')
check("Wikipedia = Encyclopedia", 'Encyclopedia' in bd, f"badge={bd}")

sc, bd = r.evaluate_source_credibility('fakewikipedia.org')
check("fakewikipedia.org blocked", 'Encyclopedia' not in bd, f"badge={bd}")

# ===== GAP 3: Empty Claims Score =====
print("\n" + "="*60)
print("GAP 3: Empty Claims = 0.0 Grounding Score")
print("="*60)

results, score = v.verify_claims([], [], [])
check("Empty claims = 0.0", score == 0.0, f"got {score}")

# ===== GAP 4: Safe Defaults =====
print("\n" + "="*60)
print("GAP 4: Model Defaults (FactualClaim + ResearchReport)")
print("="*60)

fc = FactualClaim(claim_text='test')
check("FactualClaim default = unverified", fc.verification_status == 'unverified', f"got {fc.verification_status}")
check("FactualClaim confidence = 0.0", fc.confidence_score == 0.0, f"got {fc.confidence_score}")
check("FactualClaim entailment = none", fc.entailment_type == 'none', f"got {fc.entailment_type}")

rr = ResearchReport(query='test', executive_summary='x', detailed_synthesis='x')
check("ResearchReport grounding = -1.0", rr.grounding_score == -1.0, f"got {rr.grounding_score}")

# ===== GAP 5: Compound Claims =====
print("\n" + "="*60)
print("GAP 5: Compound Claim Numbers (split across sentences)")
print("="*60)

chunk1 = RelevantChunk(source_index=1, title='AI', url='http://x.com', domain='x.com',
    text='Global AI spending is projected to reach 2.5 trillion in 2026. Generative AI spending is projected to reach 143 billion globally by end of 2026.',
    relevance_score=2.0, credibility_badge='x', credibility_score=60)
cit1 = SourceCitation(id=1, title='AI', url='http://x.com', domain='x.com', snippet='x', credibility_badge='x', credibility_score=60)
compound = FactualClaim(
    claim_text='Global AI spending is projected to reach 2.5 trillion in 2026, with generative AI spending alone hitting 143 billion [1]',
    cited_source_ids=[1], verification_status='unverified')
results, score = v.verify_claims([compound], [chunk1], [cit1])
check("Compound claim NOT unsupported", results[0].verification_status != 'unsupported',
    f"got {results[0].verification_status}: {results[0].reasoning[:100]}")

# ===== GAP 6: Dollar Sign Rendering =====
print("\n" + "="*60)
print("GAP 6: Dollar Sign LaTeX Rendering Fix")
print("="*60)

with open('app.py', 'r', encoding='utf-8') as f:
    app_src = f.read()
check("safe_md() defined in app.py", 'def safe_md' in app_src)
check("safe_md applied to executive_summary", 'safe_md(report.executive_summary)' in app_src)
check("safe_md applied to detailed_synthesis", 'safe_md(report.detailed_synthesis)' in app_src)
check("safe_md applied to consensus_analysis", 'safe_md(report.consensus_analysis)' in app_src)
check("safe_md applied to key findings", 'safe_md(finding)' in app_src)

# ===== GAP 7: Report Duplication =====
print("\n" + "="*60)
print("GAP 7: Report Section Duplication Fix")
print("="*60)

with open('src/synthesizer.py', 'r', encoding='utf-8') as f:
    syn_src = f.read()
# Count occurrences — old code had 'detailed_body = text'
lines = syn_src.split('\n')
has_old = any('detailed_body = text' in line and '#' not in line.split('detailed_body')[0] for line in lines)
check("No 'detailed_body = text' (old duplication bug)", not has_old)
check("Extracts In-Depth Analysis section only", 'In-Depth Analysis' in syn_src)

# ===== GAP 8: Freshness Date Types =====
print("\n" + "="*60)
print("GAP 8: Freshness Summary Uses Date Provenance")
print("="*60)

check("Uses date_type in freshness", 'date_type' in syn_src and 'c.date_type' in syn_src)
check("Has page-creation label", 'page_created' in syn_src or 'page-creation' in syn_src)
# Check the old misleading text is gone from the freshness function
freshness_func = syn_src[syn_src.index('def _summarize_freshness'):]
freshness_func = freshness_func[:freshness_func.index('\n    def ') if '\n    def ' in freshness_func else len(freshness_func)]
check("No 'verified publication dates' in freshness (old bug)",
    'contain verified publication dates' not in freshness_func)

# ===== SUMMARY =====
print("\n" + "="*60)
total = passed + failed
print(f"RESULTS: {passed}/{total} checks passed, {failed} failed")
if failed == 0:
    print("ALL GAPS VERIFIED - EVERYTHING IS FIXED!")
else:
    print(f"WARNING: {failed} check(s) still failing")
print("="*60)
sys.exit(0 if failed == 0 else 1)
