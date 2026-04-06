from pathlib import Path
import pandas as pd

if __name__=='__main__':
    root=Path(__file__).resolve().parent.parent
    idx=pd.read_excel(root.parent/'tablesIndex.xlsx')
    desc=pd.read_excel(root.parent/'tablesDescription.xlsx')
    print('Filas schema:', len(idx))
    print('Filas descripciones:', len(desc))
    print('Catálogo en', root/'catalog')
