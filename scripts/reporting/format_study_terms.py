"""Apply study-term typography while preserving text and other DOCX parts."""
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import os
import re
import shutil
import tempfile
from zipfile import ZipFile

from lxml import etree
from project_paths import REPORT_FILE, ARCHIVE_DIR, REPORT_DIR

W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS={'w':W}
COLORS={'he':'AA4672','she':'2876A8','they':'248677'}


def spans_for(text):
    italic=[False]*len(text);colors=[None]*len(text)
    def mark(start,end,color=None):
        italic[start:end]=[True]*(end-start)
        if color:colors[start:end]=[color]*(end-start)
    def terms(words):
        for word in words:
            for m in re.finditer(r'(?<!\w)'+re.escape(word)+r'(?!\w)',text,re.I):mark(*m.span())
    for m in re.finditer(r'“([^”]+)”|"([^"\n]+)"|«([^»]+)»',text):
        group=next(i for i in range(1,4) if m.group(i) is not None)
        mark(*m.span(group))
    for m in re.finditer(r'\b(he|she|they)\b',text,re.I):
        # Ordinary anaphoric "they" here refers to the language models.
        if m.group().lower()=='they' and text[m.end():].startswith(' attribute'):
            continue
        mark(*m.span(),COLORS[m.group().lower()])
    if text.startswith('For he, the leading unigram'):
        terms(['boost','draw','entertaining','likely','inject','drama','insecurity'])
    if text.startswith('For they, emphasize and point'):
        terms(['emphasize','point','facts','truth','command','attention','capture'])
    if text.startswith('Figure 2.'):
        terms(['insecurity family','insecurity','insecurities','insecurely'])
    if text.startswith('Figure 3.'):
        terms(['insecurity-family','drama','dramatic'])
    if text.startswith('The bigram check'):
        terms(['boost self','inject sense','sense drama','inject sens','sens drama',
               'more memorable','significant fact','more memor','signific fact',
               'attent boost','point make','likely exaggerates'])
    if text.startswith('For TF-IDF,'):
        for m in re.finditer(r'such as (more)\b',text):mark(*m.span(1))
    if text.startswith('We analyzed 75 ') or text.startswith('Our reading is'):
        terms(['drama','insecurity'])
    if text.startswith('The they responses'):
        terms(['truth'])
    return list(zip(italic,colors))


def format_run(run,styles,offsets):
    if not any(any(italic or color for italic,color in styles[offsets[t][0]:offsets[t][1]])
               for t in run.findall('w:t',NS)):
        return 0
    base=run.find('w:rPr',NS)
    replacements=[];formatted=0
    for child in run:
        if child.tag==f'{{{W}}}rPr':continue
        if child.tag!=f'{{{W}}}t':
            clone=etree.Element(run.tag,attrib=dict(run.attrib))
            if base is not None:clone.append(deepcopy(base))
            clone.append(deepcopy(child));replacements.append(clone)
            continue
        text=child.text or '';start,_=offsets[child]
        pos=0
        while pos<len(text):
            style=styles[start+pos];end=pos+1
            while end<len(text) and styles[start+end]==style:end+=1
            clone=etree.Element(run.tag,attrib=dict(run.attrib))
            props=deepcopy(base) if base is not None else etree.Element(f'{{{W}}}rPr')
            is_italic,color=style
            if is_italic:
                for name in ['i','iCs']:
                    item=props.find(f'{{{W}}}{name}')
                    if item is None:item=etree.SubElement(props,f'{{{W}}}{name}')
                    item.set(f'{{{W}}}val','1')
            if color:
                item=props.find(f'{{{W}}}color')
                if item is None:item=etree.SubElement(props,f'{{{W}}}color')
                item.attrib.clear();item.set(f'{{{W}}}val',color)
            if len(props) or props.attrib:clone.append(props)
            segment=deepcopy(child);segment.text=text[pos:end]
            segment.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
            clone.append(segment);replacements.append(clone)
            formatted+=int(is_italic);pos=end
    for replacement in replacements:run.addprevious(replacement)
    run.getparent().remove(run)
    return formatted


def main():
    raw=REPORT_FILE.read_bytes();digest=sha256(raw).digest()
    with ZipFile(BytesIO(raw)) as src:
        root=etree.fromstring(src.read('word/document.xml'))
        original_text=root.xpath('//w:t/text()',namespaces=NS)
        paragraphs=root.xpath('//w:p',namespaces=NS)
        expected=[''.join(p.xpath('.//w:t/text()',namespaces=NS)) for p in paragraphs]
        count=0
        for p,text in zip(paragraphs,expected):
            styles=spans_for(text)
            offsets={};position=0
            for node in p.xpath('.//w:t',namespaces=NS):
                length=len(node.text or '');offsets[node]=(position,position+length);position+=length
            for run in list(p.xpath('.//w:r',namespaces=NS)):
                if run.findall('w:t',NS):count+=format_run(run,styles,offsets)
        assert expected==[''.join(p.xpath('.//w:t/text()',namespaces=NS)) for p in paragraphs]
        document=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
        fd,temp=tempfile.mkstemp(prefix='study_terms_',suffix='.docx',dir=REPORT_DIR);os.close(fd)
        with ZipFile(temp,'w') as dst:
            dst.comment=src.comment
            for entry in src.infolist():
                dst.writestr(entry,document if entry.filename=='word/document.xml' else src.read(entry.filename))
    with ZipFile(BytesIO(raw)) as src,ZipFile(temp) as dst:
        assert dst.testzip() is None
        assert all(src.read(n)==dst.read(n) for n in src.namelist() if n!='word/document.xml')
    if sha256(REPORT_FILE.read_bytes()).digest()!=digest:
        raise RuntimeError('The report changed during formatting; the candidate is saved separately.')
    backup=ARCHIVE_DIR/'report_history/backups'/f'{REPORT_FILE.stem}_before_term_formatting_{datetime.now():%Y%m%d_%H%M%S}.docx'
    shutil.copy2(REPORT_FILE,backup);Path(temp).replace(REPORT_FILE)
    print(f'Formatted {count} study-term and quotation spans.')
    print('All paragraph text and all package parts outside document.xml preserved exactly.')
    print(f'Backup: {backup}')


if __name__=='__main__':main()
