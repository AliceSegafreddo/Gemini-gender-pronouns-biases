"""One-time, lossless migration of the original flat workspace layout.

Do not rerun on an already migrated project. The manifest records file hashes
and destinations; Git staging and commits are intentionally outside this tool.
"""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MOVES={
 'correct_conjugation_he_she_they.txt':'data/main/responses.txt',
 'final_analysis_gemini_prompts_responses.txt':'data/experiments/full_prompt_set/responses.txt',
 'final_he_she_they_analysis_gemini_prompts_responses.txt':'data/experiments/conjugation_comparison/responses.txt',
 'altered_conjugation_he_she_they.txt':'data/experiments/altered_conjugation/responses.txt',
 'he_she_correct_they_altered.txt':'data/experiments/he_she_correct_they_altered/responses.txt',
 'he_she_altered_they_correct.txt':'data/experiments/he_she_altered_they_correct/responses.txt',
 'all_prounouns_gemini_prompts_responses.txt':'data/experiments/all_pronouns/responses.txt',
 'proper_subject_gemini_prompts_responses.txt':'data/experiments/proper_subject/responses.txt',
 'he_she_they_gemini_prompts_responses.txt':'data/experiments/pilot_he_she_they/responses.txt',
 'images/correct_conjugation':'results/main/embeddings',
 'images/altered_conjugation':'results/experiments/altered_conjugation/embeddings',
 'images/combined_conjugations':'results/experiments/combined_conjugations/embeddings',
 'images/he_she_correct_they_altered':'results/experiments/he_she_correct_they_altered/embeddings',
 'images/he_she_altered_they_correct':'results/experiments/he_she_altered_they_correct/embeddings',
 'images/proper_subject_all-MiniLM-L12-v2':'results/experiments/proper_subject/embeddings/all-MiniLM-L12-v2',
 'images/all_pronouns_all-MiniLM-L12-v2':'results/experiments/all_pronouns/embeddings/all-MiniLM-L12-v2',
 'images/he_she_they_all-MiniLM-L12-v2':'results/experiments/pilot_he_she_they/embeddings/all-MiniLM-L12-v2',
 'images/all-MiniLM-L12-v2':'archive/figures/earlier_model_comparisons/all-MiniLM-L12-v2',
 'images/all-MiniLM-L6-v2':'archive/figures/earlier_model_comparisons/all-MiniLM-L6-v2',
 'images/bge-small-en-v1.5':'archive/figures/earlier_model_comparisons/bge-small-en-v1.5',
 'images/paraphrase-MiniLM-L3-v2':'archive/figures/earlier_model_comparisons/paraphrase-MiniLM-L3-v2',
 'images/legacy':'archive/figures/legacy',
 'report/Gemini_pronoun_analysis_report.docx':'reports/main/Gemini_pronoun_analysis_report.docx',
 'report/figures':'reports/main/figures',
 'report/backups':'archive/report_history/backups',
 'report/figure1_design_rounds':'archive/figures/figure1_design_rounds',
 'report/revisions':'archive/report_history/revisions',
 'report/introduction_revisions':'archive/report_history/introduction_revisions',
 'report/report.md':'archive/report_history/drafts/report.md',
 'report/introduction.md':'archive/report_history/drafts/introduction.md',
 'report/reference_and_calibration_additions.md':'archive/report_history/drafts/reference_and_calibration_additions.md',
 'report/Gemini_pronoun_analysis_report.docx.png':'archive/figures/report_previews/Gemini_pronoun_analysis_report.docx.png',
}
for filename in ['verified_metrics.json','response_embeddings.npz','source_hashes.json',
                 'tfidf_all_terms.csv','embedder_comparison.json','embedder_comparison.csv']:
    MOVES['report/'+filename]='results/main/report_support/'+filename
for stem,experiment in [
    ('correct_conjugation','main'),('all_prounouns','experiments/all_pronouns'),
    ('final_analysis','experiments/full_prompt_set'),
    ('final_he_she_they_analysis','experiments/conjugation_comparison'),
    ('he_she_they','experiments/pilot_he_she_they')]:
    for suffix in ['csv','xlsx']:
        MOVES[f'{stem}_tfidf_bias_terms.{suffix}']=f'results/{experiment}/tfidf/terms.{suffix}'
for name in ['report_analysis','report_embedder_comparison','refine_figure1','update_report_figure1']:
    MOVES[name+'.py']='scripts/reporting/'+name+'.py'
for name in ['analyze_combined_conjugations','filter_gemini_responses','split_conjugation_variants']:
    MOVES[name+'.py']='scripts/experiments/'+name+'.py'
for name in ['build_scientific_report','update_report_reference_and_calibration',
             'revise_report_introduction','remove_report_conjugation_mentions','remove_report_footers']:
    MOVES[name+'.py']='archive/code/report_history/'+name+'.py'


def main():
    manifest=ROOT/'archive/reorganization_manifest.json'
    if manifest.exists():raise RuntimeError('Already migrated; consult archive/reorganization_manifest.json.')
    plan=[];files=[]
    for old,new in MOVES.items():
        source=ROOT/old;dest=ROOT/new
        if not source.exists():continue
        if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}')
        plan.append((source,dest))
        for file in sorted(source.rglob('*')) if source.is_dir() else [source]:
            if file.is_file():
                target=dest/file.relative_to(source) if source.is_dir() else dest
                files.append({'from':str(file.relative_to(ROOT)),
                              'to':str(target.relative_to(ROOT)),
                              'sha256':sha256(file.read_bytes()).hexdigest()})
    for source,dest in plan:
        dest.parent.mkdir(parents=True,exist_ok=True)
        source.rename(dest)
    assert all(sha256((ROOT/f['to']).read_bytes()).hexdigest()==f['sha256'] for f in files)
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({'files':files},indent=2)+'\n')
    for old in ['images','report']:
        folder=ROOT/old
        if folder.exists() and not any(folder.iterdir()):folder.rmdir()
    print(f'Moved and hash-verified {len(files)} files; no files deleted or overwritten.')
    print(f'Manifest: {manifest}')


if __name__=='__main__':main()
