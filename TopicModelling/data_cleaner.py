# pip install -r requirements.txt

import os
import sys
modulefolder = '\\'.join(os.path.dirname(os.path.realpath(__file__)).split('\\')[:-1])
sys.path.append(modulefolder)

import re
import time
import nltk
import spacy

import pandas as pd
pd.set_option("display.max_colwidth", 200)

from tqdm import tqdm
from unidecode import unidecode
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
from nltk.corpus import stopwords

stopwords_french = stopwords.words('french')
stopwords_french.extend(['nan', 'à', 'les', 'ça', 'cette', 'aussi', 'si', 'faut', 'ils', 'cela', 'car', 'comme', 'tout',
                         'peut', 'après', 'deja', 'tous', 'ni', 'la', 'là', 'ceux', 'celles', 'tout', 'toutes', 'le', 'alors',
                         'très', 'donc', 'fait', 'al', 'aux'])

stopwords_french = [unidecode(x) for x in stopwords_french]

nlp = spacy.load('fr')


class DataCleaner:
    """
    A class used to clean the csv files of our project \n
    Do not forget to run 'pip install -r requirements.txt' to avoid any missing packages errors \n
    Run in a python terminal : \n
    `import ntlk` \n
    `nltk.download('stopwords')` for stopwords \n
    Then run in a terminal : \n
    `python -m spacy download fr` for lemmatization
    """

    def __init__(self, folder, filename, force=False, user=False, lemma=False):
        """
        Initialize the DataCleaner \n\n
        `folder`: the path of the folder containing the data \n
        `filename`: the file to clean \n
        NOTE : if `filename == 'all' `, all the csv files not starting with 'clean' or 'QUESTIONNAIRE' will be clean
        `force`: re-run the cleaner on files are already cleaned \n
        `user`: merge the 4 datasets inside one grouped by users' id \n
        `lemma`: to clean with or without lemmatization of the words, by using the package Spacy (VERO LONG COMPUTING
        TIME) \n
        """
        self.folder = folder
        self.filename = filename

        self.force = force
        self.user = user
        self.lemma = lemma

    @staticmethod
    def clean_doc(doc, lemma=False):
        """
        Clean a string containing an answer \n\n
        `doc`: a string \n
        `lemma`: to keep only the lemma of the word of not -> computer intensive \n
        """
        if not lemma:
            # removing accent
            doc = unidecode(doc)
            # "80km/h 80 km/h 80 kmh" --> "80km/h 80km/h 80km/h"
            doc = re.sub(r'80 ?km/?h', '80kmh', doc)
            # removing everything except alphabets
            doc = re.sub(r'[^a-zA-Z0-9]', ' ', doc)
            # make all text lowercase
            doc = doc.lower()
            # remove stopwords
            tokenized_doc = doc.split()
            tokenized_doc = [word for word in tokenized_doc if word not in stopwords_french]
            clean_doc = ' '.join(tokenized_doc)
        else:
            # "80km/h 80 km/h 80 kmh" --> "80km/h 80km/h 80km/h"
            doc = re.sub(r'80 ?km/?h', '80kmh', doc)
            # make all text lowercase
            doc = doc.lower()
            # Lemmatization
            tokenized_doc = nlp(doc)
            tokenized_doc = [word.lemma_ for word in tokenized_doc]
            clean_doc = ' '.join(tokenized_doc)
            # # removing accent
            clean_doc = unidecode(clean_doc)
            # removing everything except alphabets
            clean_doc = re.sub(r'[^a-zA-Z0-9]', ' ', clean_doc)
            # remove stopwords
            tokenized_doc = clean_doc.split()
            tokenized_doc = [word for word in tokenized_doc if word not in stopwords_french]
            clean_doc = ' '.join(tokenized_doc)

        return clean_doc

    def clean_col(self, df, col_name):
        """
        Clean one column of the dataset by making all text lowercase and removing : \n
        - accents \n
        - everything except alphabets \n
        - apostrophe character \n
        - short words \n
        - stopwords \n\n
        `df`: the DataFrame containing the dataset \n
        `col_name`: a string containing the name of the column to clean \n
        `return`: the column cleaned \n
        """
        column = df[col_name].fillna('')

        clean_column = column.apply(lambda x: DataCleaner.clean_doc(x, self.lemma))

        return clean_column

    def detect_open_answer(self, df, col_name):
        """
        Detect if the answer is open or closed \n
        In Topic Modelling, we want to analyse only the open answers \n\n
        `df`: the dataframe containing the dataset \n
        `col_name`: the colname to analyse \n
        `return`: a boolean if the answer is open (True) or closed (False) \n
        """
        col = df[col_name].fillna('')
        answers = []
        for i in range(len(col)):
            if col[i] not in answers:
                answers.append(col[i])
            if len(answers) > 5:
                return True  # It is an open question
            if i > 100:
                return False  # It is a closed question (Yes/No/IdK)

    def column_to_keep(self, df):
        """
        Determine the column to keep in the cleaned dataframe : only the open answers \n\n
        `df`: the dataframe containing the dataset \n
        `return`: a list containing the names of the columns to keep for analyse \n
        """
        col_to_keep = []
        col_to_keep.extend(['authorId', 'authorZipCode', 'title'])
        col_to_keep.extend([colname for colname in df.columns if 'QUXV' in colname])
        return col_to_keep

    def column_to_keep_with_user(self, df):
        """
        Determine the column to keep in the cleaned dataframe : only the open answers \n\n
        `df`: the dataframe containing the dataset \n
        `return`: a list containing the names of the columns to keep for analyse \n
        """
        col_to_keep = ["authorId"]
        col_to_keep.extend(["title"])
        col_to_keep.extend([colname for colname in df.columns if 'QUXV' in colname])
        return col_to_keep

    def clean_dataframe(self, df):
        """
        Clean the dataframe \n\n
        `df`: the dataframe containing the dataset \n
        `return`: a new dataframe containing only the columns to keep cleaned \n
        """
        data = {}
        col_to_keep = self.column_to_keep(df)
        col_to_add_but_not_clean = ['authorId', 'authorZipCode']

        for i in tqdm(range(len(col_to_keep))):
            col_name = col_to_keep[i]
            if col_name in col_to_add_but_not_clean and col_name in df.columns:
                data[col_name] = df[col_name]
            else:
                if self.detect_open_answer(df, col_name):
                    col_clean = self.clean_col(df, col_name)
                    data[col_name] = col_clean

        return pd.DataFrame(data=data)

    def clean_dataframe_with_user(self, df):
        """
        Clean the dataframe \n\n
        `df`: the dataframe containing the dataset \n
        `return`: a new dataframe containing only 2 columns : AuthorID et Documents \n
        Each rows contains a string with all the answers a person made across the 4 datasets \n
        """
        data = {}
        col_to_keep = self.column_to_keep_with_user(df)

        for i in tqdm(range(len(col_to_keep))):
            col_name = col_to_keep[i]
            if self.detect_open_answer(df, col_name):
                col_clean = self.clean_col(df, col_name)
                data[col_name] = col_clean
                
        documents = df[col_to_keep[2]].fillna('').map(str)
        for i in range(3, len(col_to_keep)):
            documents = documents + ' ' + df[col_to_keep[i]].fillna('').map(str)
            
        data['documents'] = documents
            
        return pd.DataFrame(data=data)[['authorId', 'documents']]

    def clean_file(self, filename):
        """
        Clean a file and write a new one under the name "clean_"+filename inside the same folder as the file \n\n
        `filename`: the file to clean \n
        """
        clean_file_path = os.path.join(self.folder, "clean_" + filename)
        if os.path.isfile(clean_file_path) and not self.force:
            print(filename + " is already cleaned!")
        else:
            df = pd.read_csv(os.path.join(self.folder, filename))
            print(filename + " read")
            df_clean = self.clean_dataframe(df)
            print(filename + " cleaned")
            df_clean.to_csv(clean_file_path, encoding="utf-8")
            print("clean_" + filename + " wrote")
    
    def clean_file_with_user(self, filename):
        """
        Clean a file and write a new one under the name clean_all_user_df.csv inside the same folder as the file \n\n
        `filename`: the file to clean \n
        """

        df = pd.read_csv(os.path.join(self.folder, filename))
        print(filename + " read")
        df_clean = self.clean_dataframe_with_user(df)
        print(filename + " cleaned")
        
        return df_clean

    def concat_answers(self, filename):
        """
        Concat the open answers of a dataset inside a single columns \n\n
        `filename`: the filename of the dataset \n
        `return`: a dataframe with a new column 'documents' \n
        """
        df = pd.read_csv(os.path.join(self.folder, "clean_" + filename))
        col_to_keep = self.column_to_keep(df)
        if 'authorId' in col_to_keep:
            col_to_keep.remove('authorId')
        if 'authorZipCode' in col_to_keep:
            col_to_keep.remove('authorZipCode')
        print('Concatening ' + filename)
        df['documents'] = df[col_to_keep].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
        print(filename + ' concatened')
        return df

    def clean(self):
        """
        Clean all the csv files not starting with 'clean' or 'QUESTIONNAIRE' \n\n
        This methods uses multithreading : each core of the local machine is used to clean a different file,
        in order to same time. \n\n
        """
        filenames = [f for f in listdir(self.folder) if isfile(join(self.folder, f)) and '.csv' in f]
        filenames = [f for f in filenames if 'clean' not in f and 'QUESTIONNAIRE' not in f]
        if self.filename == 'all':
            print('File to clean : ', end="")
            print(filenames)
            pool = Pool()
            file_cleaners = pool.map(self.clean_file, filenames)
            pool.terminate()
            del pool
        else :
            self.clean_file(self.filename)
        print('Files Cleaned!')
        if self.user:
            filename_clean_user = 'clean_all_user_df.csv'
            if not self.force:
                if os.path.isfile(os.path.join(self.folder, filename_clean_user)):
                    print('Users File already cleaned !')
                    return
            self.force = False
            pool = Pool()
            file_cleaners = pool.map(self.clean_file, filenames)
            pool.terminate()
            del pool
            pool = Pool()
            dfs_concat = pool.map(self.concat_answers, filenames)
            pool.terminate()
            del pool
            all_df_clean = pd.concat(dfs_concat, join='outer')
            print('Grouping the dataframes by authorId')
            all_df_clean = all_df_clean.groupby('authorId')['documents'].apply(
                lambda x: ' '.join(x.astype(str))).reset_index()
            print('Dataframes Grouped')
            print('Writting the dataframe to csv')
            all_df_clean.to_csv(os.path.join(self.folder, filename_clean_user), encoding='utf-8')
            print('File wrote')


if __name__ == '__main__':
    folder = "../../data"
    filename = "DEMOCRATIE_ET_CITOYENNETE.csv"
    dataCleaner = DataCleaner(folder, filename, force=True, user=False, lemma=False)
    start = time.time()
    dataCleaner.clean()
    end = time.time()
    print("Cleaned in %f min" % ((end-start)/60))
