"""Pipeline test script — quick end-to-end validation."""

import sys
sys.path.insert(0, '.')

import yaml
from src.pipeline import PodAgentPipeline

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Create pipeline
pipeline = PodAgentPipeline(config=config)

# Test with a known podcast ID
result = pipeline.analyze_existing(2)
if result:
    print(f"Podcast ID: {result['podcast_id']}")
    print(f"Summary: {result['summary_path']}")
    print(f"Insights: {result['insights_path']}")
else:
    print("No existing podcast found for analysis")

print("Pipeline validation complete")
