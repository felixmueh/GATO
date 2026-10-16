#!/usr/bin/env python3
"""Render this campaign's Markdown text, tables and embedded figures to PDF.

Requires Matplotlib and Pillow. Uses relative report figure paths; no raw data.
Example: python report_to_pdf.py --report REPORT.md --previews /tmp/report-preview
"""
import argparse
from pathlib import Path
import re
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def render(report,previews=None):
    report=Path(report).resolve();lines=report.read_text().splitlines();page=0
    if previews:previews=Path(previews);previews.mkdir(parents=True,exist_ok=True)
    with PdfPages(report.with_suffix('.pdf')) as pdf:
        fig=None;y=0;code=False
        def save():
            nonlocal fig,page
            if fig is None:return
            page+=1;fig.text(.95,.025,str(page),ha='right',fontsize=8)
            pdf.savefig(fig)
            if previews:fig.savefig(previews/f'page-{page:02d}.png',dpi=110)
            plt.close(fig);fig=None
        def new():
            nonlocal fig,y
            save();fig=plt.figure(figsize=(8.27,11.69));y=.94
        def text(value,heading=0,mono=False):
            nonlocal y
            if not value.strip():
                if fig is not None:y-=.012
                return
            for row in textwrap.wrap(value,width=112 if mono else 91,break_long_words=True,replace_whitespace=False) or ['']:
                if fig is None or y<.075:new()
                fig.text(.075,y,row,va='top',fontsize=15 if heading==1 else 12 if heading else 8 if mono else 10,
                    fontweight='bold' if heading else 'normal',family='monospace' if mono else 'sans-serif')
                y-=.026 if heading else .018 if mono else .022
        def table(rows):
            """Draw wrapped cells and repeat the header when a table spans pages."""
            nonlocal y
            clean=lambda value:value.strip().replace('**','').replace('`','')
            cells=[[clean(cell) for cell in row.strip().strip('|').split('|')] for row in rows]
            header=cells[0];body=cells[1:];columns=len(header)
            if any(len(row)!=columns for row in body):
                raise ValueError('Inconsistent Markdown table column count')
            sizes=[min(30,max(7,max(len(row[j]) for row in cells))) for j in range(columns)]
            widths=[size/sum(sizes) for size in sizes]
            font=7.2 if columns>=8 else 8
            budgets=[max(5,int(width*(112 if font<8 else 92))) for width in widths]
            wrapped=[['\n'.join(textwrap.wrap(cell,width=budgets[j],break_long_words=True) or [''])
                for j,cell in enumerate(row)] for row in cells]
            heights=[.014*max(cell.count('\n')+1 for cell in row)+.012 for row in wrapped]
            index=1
            while index<len(wrapped):
                if fig is None or y-heights[0]-heights[index]<.075:new()
                end=index;height=heights[0]
                while end<len(wrapped) and y-height-heights[end]>=.075:
                    height+=heights[end];end+=1
                if end==index:
                    raise ValueError('Table row is too tall for one PDF page')
                shown=[wrapped[0],*wrapped[index:end]]
                row_heights=[heights[0],*heights[index:end]]
                ax=fig.add_axes([.075,y-height,.85,height]);ax.axis('off')
                drawn=ax.table(cellText=shown,colWidths=widths,cellLoc='left',bbox=[0,0,1,1])
                drawn.auto_set_font_size(False);drawn.set_fontsize(font)
                for (i,j),cell in drawn.get_celld().items():
                    cell.set_height(row_heights[i]/height);cell.set_edgecolor('#b8b8b8');cell.set_linewidth(.4)
                    cell.PAD=.07
                    if i==0:cell.set_facecolor('#e9edf0');cell.set_text_props(weight='bold')
                    elif i%2==0:cell.set_facecolor('#f6f6f6')
                y-=height+.02;index=end
                if index<len(wrapped):new()
        index=0
        while index<len(lines):
            line=lines[index];index+=1
            if not code and line.startswith('|'):
                rows=[line]
                while index<len(lines) and lines[index].startswith('|'):
                    if not re.fullmatch(r'\|[\s:|\-]+\|',lines[index]):rows.append(lines[index])
                    index+=1
                table(rows);continue
            image=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line.strip())
            if image:
                save();fig=plt.figure(figsize=(11.69,8.27));ax=fig.add_axes([.055,.20,.89,.75]);ax.imshow(plt.imread(report.parent/image[2]));ax.axis('off')
                fig.text(.055,.135,textwrap.fill(image[1],125),fontsize=10,va='top');save();continue
            if line.startswith('```'):code=not code;continue
            if re.fullmatch(r'\|[\s:|\-]+\|',line):continue
            heading=len(line)-len(line.lstrip('#')) if line.startswith('#') else 0
            value=line.lstrip('#').strip() if heading else line
            value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1 (\2)',value)
            value=value.replace('**','').replace('`','')
            text(value,heading=min(heading,2),mono=code or line.startswith('|'))
        save()
    return page

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,default=Path('REPORT.md'));p.add_argument('--previews',type=Path);a=p.parse_args()
    print(render(a.report,a.previews),'PDF pages')
