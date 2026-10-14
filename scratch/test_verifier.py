import sys
sys.path.append(r'C:/Users/deepi/Downloads/ai-powered-web-research')
from src.pipeline import WebResearchPipeline

pipeline = WebResearchPipeline(groq_api_key='')
query = "When did NASA's Perseverance rover land on Mars? Give the exact date and time in UTC, cite official NASA source."
report, ctx, docs, chunks = pipeline.run_research(query=query, depth='quick', max_sources=5)
print('Grounding Score:', report.grounding_score)
for claim in report.claims:
    print('Claim:', claim.claim_text)
    print('  Status:', claim.verification_status)
    print('  Confidence:', claim.confidence_score)
    print('  Excerpt:', claim.supporting_excerpt)
    print('---')
