"""Replace only Figure 1's media bytes, preserving every other DOCX member."""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import os
import shutil
import tempfile
from zipfile import ZipFile

from docx import Document
from project_paths import REPORT_FILE, REPORT_DIR, ARCHIVE_DIR


def main():
    report=REPORT_FILE
    replacement=REPORT_DIR/'figures/distances.png'
    original_hash=sha256(report.read_bytes()).digest()
    doc=Document(report)
    first_picture=doc.inline_shapes[0]
    rel_id=first_picture._inline.graphic.graphicData.pic.blipFill.blip.embed
    member=str(doc.part.related_parts[rel_id].partname).lstrip('/')
    backup_dir=ARCHIVE_DIR/'report_history/backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup=backup_dir/(report.stem+'_before_figure1_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.docx')
    shutil.copy2(report,backup)
    fd,path=tempfile.mkstemp(prefix='figure1_',suffix='.docx',dir=report.parent)
    os.close(fd)
    temp=Path(path)
    with ZipFile(report) as src,ZipFile(temp,'w') as dst:
        dst.comment=src.comment
        for entry in src.infolist():
            dst.writestr(entry,replacement.read_bytes() if entry.filename==member else src.read(entry.filename))
    with ZipFile(report) as before,ZipFile(temp) as after:
        changed=[n for n in before.namelist() if before.read(n)!=after.read(n)]
        assert changed==[member],changed
        assert after.testzip() is None
    assert len(Document(temp).inline_shapes)==len(doc.inline_shapes)
    if sha256(report.read_bytes()).digest()!=original_hash:
        raise RuntimeError('The report changed during the update; the new image is saved separately for a retry.')
    temp.replace(report)
    print(f'Updated only {member}. Every other DOCX member is byte-for-byte identical.')
    print(f'Backup: {backup}')


if __name__=='__main__':main()
