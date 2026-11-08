"""Regenerate the three pronoun-colored plots and replace their Word images."""
import csv
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

from PIL import Image

from project_paths import ARCHIVE_DIR, REPORT_DIR, REPORT_FILE, REPORT_SUPPORT
from .report_analysis import save_lexical_figures


def main():
    names=['unigram_rankings.png','lexical_counts.png','bigram_rankings.png']
    source=REPORT_FILE.read_bytes()
    digest=sha256(source).digest()
    old={name:(REPORT_DIR/'figures'/name).read_bytes() for name in names}
    # Match images by content rather than relying on figure numbering or order.
    targets={}
    with ZipFile(BytesIO(source)) as z:
        for name,data in old.items():
            matches=[n for n in z.namelist() if n.startswith('word/media/') and z.read(n)==data]
            if len(matches)!=1:
                raise RuntimeError(f'Cannot uniquely match {name} to an embedded image; report left untouched.')
            targets[matches[0]]=name
    backup=ARCHIVE_DIR/'report_history/backups'/f'pronoun_colors_{datetime.now():%Y%m%d_%H%M%S}'
    backup.mkdir(parents=True)
    shutil.copy2(REPORT_FILE,backup/REPORT_FILE.name)
    for name in names:shutil.copy2(REPORT_DIR/'figures'/name,backup/name)
    metrics=json.loads((REPORT_SUPPORT/'verified_metrics.json').read_text())
    with (REPORT_SUPPORT/'tfidf_all_terms.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:row['tf_idf']=float(row['tf_idf'])
    safe_rows=[r for group in metrics['within_response_top_bigrams'].values() for r in group]
    save_lexical_figures(rows,safe_rows,metrics)
    for name in names:
        original=Image.open(BytesIO(old[name]))
        updated=Image.open(REPORT_DIR/'figures'/name)
        assert original.size==updated.size, f'The figure dimensions changed: {name}'
    replacements={member:(REPORT_DIR/'figures'/name).read_bytes() for member,name in targets.items()}
    fd,temp=tempfile.mkstemp(prefix='pronoun_colors_',suffix='.docx',dir=REPORT_DIR)
    os.close(fd)
    with ZipFile(BytesIO(source)) as src,ZipFile(temp,'w') as dst:
        dst.comment=src.comment
        for entry in src.infolist():dst.writestr(entry,replacements.get(entry.filename,src.read(entry.filename)))
    with ZipFile(BytesIO(source)) as src,ZipFile(temp) as dst:
        changed={n for n in src.namelist() if src.read(n)!=dst.read(n)}
        assert changed==set(targets),changed
        assert dst.testzip() is None
    if sha256(REPORT_FILE.read_bytes()).digest()!=digest:
        raise RuntimeError('The report changed during the update; the figures are saved separately for a retry.')
    Path(temp).replace(REPORT_FILE)
    print('Updated only the three lexical figure images: She blue; He previous She color; They green.')
    print('All text, formatting, other figures, and image dimensions preserved.')
    print(f'Backup: {backup}')


if __name__=='__main__':main()
