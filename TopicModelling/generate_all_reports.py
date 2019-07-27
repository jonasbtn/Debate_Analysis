"""
Generate_all_report is a script used to run a LDA on all the files \n\n
First, the LDA is run for each columns of each file \n
Then, the LDA is run on the entire file \n  \n
The script uses multithreading to run 4 LDAs at the same time, one on each core of the machine, to save time \n\n

Usage : \n

- edit the folder variable to the location of the data \n

- run in a terminal : \n
python generate_all_reports.py \n \n

Do not forget to run 'pip install -r requirements.txt' to avoid any missing packages errors \n
"""

import os
import sys
modulefolder = '\\'.join(os.path.dirname(os.path.realpath(__file__)).split('\\')[:-1])
sys.path.append(modulefolder)

from TopicModelling.Topic_Modelling_LDA import LDA
from multiprocessing import Pool
import pandas as pd
import os
import time

folder = "../../data"
filenames = ["DEMOCRATIE_ET_CITOYENNETE.csv", "LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.csv",
             "LA_TRANSITION_ECOLOGIQUE.csv", "ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.csv"]


def lda_all(filename):
    lda = LDA(folder, filename, ['all'], 12, sample_only=False, display=False, report=True, cluster=False)
    lda.run()
    return lda


def lda_col(filename,colname):
    if colname == "title":
        ntopic = 8
    else :
        ntopic = 4
    lda = LDA(folder, filename, [colname], ntopic, sample_only=False, display=False, report=True, cluster=False)
    lda.run()
    return lda


def col_to_keep(df):
    columns = df.columns
    col_to_analyse = [colname for colname in columns if colname.startswith('QUXV') or colname == 'title']
    return col_to_analyse


def generate_report_col_only():
    for filename_local in filenames:
        filename = filename_local
        df = pd.read_csv(os.path.join(folder, "clean_"+filename), nrows=1)
        col_to_analyse = col_to_keep(df)
        tuples = create_tuple(filename, col_to_analyse)
        pool = Pool(processes=4)
        ldas_col_only = pool.starmap(lda_col, tuples)
        pool.terminate()
        del pool


def create_tuple(filename, col_to_analyse):
    tuples = []
    for col in col_to_analyse:
        tuples.append((filename, col))
    return tuples


def generate_report_all():
    pool = Pool(processes=4)
    ldas_all = pool.map(lda_all, filenames)
    pool.terminate()
    del pool


if __name__ == '__main__':

    start = time.time()

    generate_report_col_only()

    generate_report_all()

    end = time.time()

    print("Lda on all files in %f sec" %(end-start))
