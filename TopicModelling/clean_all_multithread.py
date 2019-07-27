"""
clean_all_multithread is a script used to preprocess all the dataset using the DataCleaner module. \n
The script uses multithread to clean the 4 files at the same time, one on each core of the machine, to save time \n\n

Usage : \n

- edit the folder variable to the location of the data \n

- run in a terminal : \n
python clean_all_multithread.py \n

Do not forget to run 'pip install -r requirements.txt' to avoid any missing packages errors \n
"""

import os
import sys
modulefolder = '\\'.join(os.path.dirname(os.path.realpath(__file__)).split('\\')[:-1])
sys.path.append(modulefolder)

import time

from TopicModelling.data_cleaner import DataCleaner
from multiprocessing import Pool


folder = "../../data"

filenames = ["DEMOCRATIE_ET_CITOYENNETE.csv", 'LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.csv', 'LA_TRANSITION_ECOLOGIQUE.csv', 'ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.csv']


def clean_file(filename):
    dataCleaner = DataCleaner(folder, filename, force=False, user=False, lemma=False)
    dataCleaner.clean()


if __name__ == '__main__':
    start = time.time()
    pool = Pool()
    dataCleaners = pool.map(clean_file, filenames)
    end = time.time()
    print("All files cleaned in %f sec" % (end-start))
    start = time.time()
    user_cleaner = DataCleaner(folder, 'all', force=False, user=True, lemma=False)
    user_cleaner.clean()
    end = time.time()
    print("Users dataset created in %f sec" % (end - start))
