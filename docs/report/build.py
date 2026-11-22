# Rebuild the report: data -> figures -> PDF. Run from the repo root:
#   python docs/report/build.py            (everything)
#   python docs/report/build.py --pdf      (only the PDF, after editing report.typ)
#   python docs/report/build.py --v2 [--pdf]  (report_v2.typ -> report_v2.pdf)
# Needs: pip install typst matplotlib   (or the typst CLI: typst compile docs/report/report.typ)
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = '--v2' in sys.argv   # report_v2: its own data and figures (make_*_v2.py); also reads v1's data.json and figures
NAME = 'report_v2' if V2 else 'report'
if '--pdf' not in sys.argv:
    for script in (['make_data_v2.py', 'make_figures_v2.py'] if V2 else ['make_data.py', 'make_figures.py']):
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)
import typst
typst.compile(os.path.join(HERE, NAME + '.typ'), output=os.path.join(HERE, NAME + '.pdf'), root=HERE,
              font_paths=['/usr/share/fonts'])
print('wrote', os.path.join(HERE, NAME + '.pdf'))
