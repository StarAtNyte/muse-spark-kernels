from pathlib import Path
import runpy
Path('dev/physical_order/sort.cu').write_text(runpy.run_path('dev/physical_order/sort.py')['CUDA'])
