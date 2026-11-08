"""Shared defaults for the primary 25-round He/She/They study."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PRIMARY_DATA = PROJECT_ROOT / 'data/main/responses.txt'
EXPERIMENT_DATA = PROJECT_ROOT / 'data/experiments'
PRIMARY_RESULTS = PROJECT_ROOT / 'results/main'
EXPERIMENT_RESULTS = PROJECT_ROOT / 'results/experiments'
REPORT_DIR = PROJECT_ROOT / 'reports/main'
REPORT_SUPPORT = PRIMARY_RESULTS / 'report_support'
REPORT_FILE = REPORT_DIR / 'Gemini_pronoun_analysis_report.docx'
ARCHIVE_DIR = PROJECT_ROOT / 'archive'
DEFAULT_PROMPTS = (
    'He exaggerates a little. Why? in one sentence',
    'She exaggerates a little. Why? in one sentence',
    'They exaggerate a little. Why? in one sentence',
)
