import os
import warnings
import logging
import datetime
import traceback


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.colors as mcolors
import wordcloud as wc

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import Label
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from TopicModelling.data_cleaner import DataCleaner


class TopicModelling:
    """
    An abstract class to run a topic modelling on a dataset to extract the most recurrent topics. \n
    THIS CLASS CANNOT BE INSTANCED DIRECTLY. \n
    To run a topic modelling, choose a model between LSA and LDA. The run it by instancing an LSA or LDA object. \n
    Do not forget to run 'pip install -r requirements.txt' to avoid any missing packages errors \n
    """

    def __init__(self, folder, filename, columns, n_topics, sample_only=True, display=False, report=False,
                 cluster=False):
        """
        Initialize the Topic Modelling object \n

        `folder`: the path of the folder containing the data \n
        `filename`: the name of the file to analyse \n
        `columns`: a list with the name of the column to analyse inside the file, ['all'] to include all the dataset \n
        `n_topics`: the number of topics to extract \n
        `sample_only`: to analyze only a sample of the dataset in case of lack of computational power (limits to
        10K observations) \n
        `display`: to display each plot in a new windows \n
        `report`: to generate a pdf report contaning all the output and graphs of the topic modelling analysis \n
        NOTE : The reports are located in the folder containing the data inside the folder "grand_debat_report" \n
        `cluster`: to create a cluster of the topics --> Computer Intensive and very long \n
        `inference`: used to analyse the dataset of all users \n
        """

        self.folder = folder
        self.filename = filename

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")

        self.logname = now + filename[:len(filename)-4]

        if self.columns == ['all']:
            self.logname += '_all'
        else:
            all_columns = list(pd.read_csv(os.path.join(self.folder, self.filename), nrows=1).columns)
            col_indices = ""
            for colname in self.columns:
                if colname in all_columns:
                    self.logname += "_" + str(all_columns.index(colname))
                else:
                    warnings.warn(colname + " not in the dataset --> Ignoring")
                    self.columns.remove(colname)

        if not os.path.isdir(os.path.join(self.folder, "grand_debat_reports")):
            os.mkdir(os.path.join(self.folder, "grand_debat_reports"))

        self.logname = os.path.join(self.folder, "grand_debat_reports", self.logname + ".log")

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.logger_file_handler = logging.FileHandler(filename=self.logname)
        self.logger_file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt='%(asctime)s-%(msecs)d %(name)s %(levelname)s %(message)s \n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger_file_handler.setFormatter(formatter)
        self.logger.addHandler(self.logger_file_handler)

        self.report = report
        if self.report:
            report_name = self.logname[:len(self.logname)-4] + "_report.pdf"
            self.pdf = PdfPages(report_name)

        if not os.path.isfile(os.path.join(folder, "clean_" + filename)):
            print("Cleaning file")
            self.logger.info("Cleaning file")
            if filename == "all_user_df.csv":
                dataCleaner = DataCleaner(folder, 'all', force=False, user=True, lemma=False)
            else:
                dataCleaner = DataCleaner(folder, filename, force=True, user=False, lemma=False)
            dataCleaner.clean()
            print("File Cleaned")
            self.logger.info("File Cleaned")

        self.sample_only = sample_only

        if self.sample_only:
            self.df = pd.read_csv(os.path.join(folder, "clean_" + filename), nrows=1000)
        else:
            self.df = pd.read_csv(os.path.join(folder, "clean_" + filename))

        self.documents = self.initiate_documents(self.df, self.columns)

        if (self.documents is None) or self.documents.empty is True:
            raise ValueError('Data Not Loaded, check self.documents')

        print('Data Loaded : %d answers to analyse in file : %s' % (len(self.documents), filename))
        self.logger.info("Data Loaded : %d answers to analyse in file : %s" % (len(self.documents), filename))

        print('Analysing column : ' + str(columns))
        self.logger.info('Analysing column : ' + str(columns))

        self.n_topics = n_topics

        self._display = display
        self._cluster = cluster

        if not self._display:
            plt.ioff()

        print('Preprocessing the data')
        self.logger.info('Preprocessing the data')
        self.preprocessing()
        print('Preprocessing Done')
        self.logger.info("Preprocessing Done")

        # Variable to be initiated later
        # self.model
        # self.topic_matrix
        # self.keys
        # self.count_vectorizer
        # self.document_term_matrix
        # self.categories
        # self.counts
        # self.top_words
        # self.mean_topic_vectors


    @staticmethod
    def initiate_documents(df, columns):
        """
        `df`: the dataframe \n
        `columns`: the columns to add \n
        `return` a list of documents \n
        """
        # initiating the list of documents
        # if multiple columns, the columns are concatenated

        all_columns = df.columns.tolist()
        if type(columns) is not list:
            raise ValueError('columns must be a list!')
        if columns == ['all']:
            columns_to_load = [colname for colname in all_columns if colname == 'title' or colname.startswith('QUXV')]
        else:
            columns_to_load = columns
        for col in columns_to_load:
            if col not in all_columns:
                warnings.warn(col + ' is not in the documents --> ignored')
                columns_to_load.remove(col)
        documents = df[columns_to_load[0]].fillna('').map(str)
        for i in range(1, len(columns_to_load)):
            documents = documents + ' ' + df[columns_to_load[i]].fillna('').map(str)
        return documents

    def preprocessing(self):
        """
        Initiate the document term matrix
        """
        self.count_vectorizer = CountVectorizer(stop_words='english')
        self.document_term_matrix = self.count_vectorizer.fit_transform(self.documents.astype('U'))

    def get_keys(self, topic_matrix):
        """
        returns an integer list of predicted topic categories for a given topic matrix \n
        For example : topic_matrix[0] = [0.04166669, 0.04166669, 0.04166669, 0.70833314, 0.04166669,
       0.0416667 , 0.0416667 , 0.0416667 ] means that the first document belongs to the topic 3 with the highest probability \n

        `topic_matrix`: a topic_matrix \n
        `return` an integer list \n
        """
        keys = []
        for i in range(topic_matrix.shape[0]):
            keys.append(topic_matrix[i].argmax())

        self.keys = keys

    def keys_to_counts(self):
        """
        the get_keys method has to be executed at least once to run this method \n

        `return` returns a tuple of topic categories and their accompanying magnitudes for the list of keys \n
        """
        count_pairs = Counter(self.keys).items()
        self.categories = [pair[0] for pair in count_pairs]
        self.counts = [pair[1] for pair in count_pairs]

    def get_top_n_words(self, n, keys, document_term_matrix, count_vectorizer):
        """
        `n`: number of top words to compute for each topic \n
        `keys`:  an integer list obtaining with the method get_keys \n
        `document_term_matrix`: a document/term matrix obtaing with a CountVectorizer \n
        `count_vectorizer`: a CountVectorizer object used to create the document term matrix \n
        `return` returns a list of n_topic strings, where each string contains the n most common
        words in a predicted category, in order \n
        """
        top_word_indices = []
        for topic in range(self.n_topics):
            temp_vector_sum = 0
            found = False
            for i in range(len(keys)):
                if keys[i] == topic:
                    temp_vector_sum += document_term_matrix[i]
                    found = True
            if found:
                temp_vector_sum = temp_vector_sum.toarray()
                top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:], 0)
                top_word_indices.append(top_n_word_indices)
            else:
                top_word_indices.append([])
        self.top_words = []
        for topic in top_word_indices:
            topic_words = []
            for index in topic:
                temp_word_vector = np.zeros((1, document_term_matrix.shape[1]))
                temp_word_vector[:, index] = 1
                the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
                topic_words.append(the_word.encode('ascii').decode('utf-8'))
            self.top_words.append(" ".join(topic_words))

    def display_top_n_word(self):
        """
        Display a bar chart with the number of documents for each topic \n
        """
        for i in range(len(self.top_words)):
            print("Topic {}: ".format(i), self.top_words[i])
            self.logger.info("Topic %d: %s" % (i, self.top_words[i]))

        labels = ['Topic {}: \n'.format(i) + ' '.join(self.top_words[i].split(' ')[0:2]) for i in self.categories]

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.bar(self.categories, self.counts)
        ax.set_xticks(self.categories)
        ax.set_xticklabels(labels)
        ax.set_title('Topic Category Counts')

        if self.report:
            self.pdf.savefig(fig)

        if self._display:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def dimensional_reduction_tsne(topic_matrix):
        """
        Reduce the dimension of the topic matrix to 2D in order to display the cluster of a graph \n\n
        `topic_matrix`: a topic matrix \n
        `return` a 2D vector of the topic matrix \n
        """
        tsne_model = TSNE(n_components=2, perplexity=50, learning_rate=100,
                          n_iter=2000, verbose=1, random_state=0, angle=0.75)
        tsne_vectors = tsne_model.fit_transform(topic_matrix)
        return tsne_vectors

    def get_mean_topic_vectors(self, keys, two_dim_vectors):
        """
        `keys`: a list of the topics of each documents\n
        `two_dim_vectors`: a two dimensional vector reduced by tsne \n
        `return` a list of centroid vectors from each predicted topic category \n
        """
        mean_topic_vectors = []
        for t in range(self.n_topics):
            articles_in_that_topic = []
            for i in range(len(keys)):
                if keys[i] == t:
                    articles_in_that_topic.append(two_dim_vectors[i])

            articles_in_that_topic = np.vstack(articles_in_that_topic)
            mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
            mean_topic_vectors.append(mean_article_in_that_topic)
        self.mean_topic_vectors = mean_topic_vectors
        return mean_topic_vectors

    def display_cluster(self, tsne_vectors):
        # output_notebook()
        colormap = np.array([
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
            "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
            "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
            "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])
        colormap = colormap[:self.n_topics]

        plot = figure(title="t-SNE Clustering of {} LSA Topics".format(self.n_topics), plot_width=1000, plot_height=1000)
        plot.scatter(x=tsne_vectors[:, 0], y=tsne_vectors[:, 1], color=colormap[self.keys])

        for t in range(self.n_topics):
            label = Label(x=self.mean_topic_vectors[t][0], y=self.mean_topic_vectors[t][1],
                          text=' '.join(self.top_words[t].split(' ')[0:2]),
                          text_color=colormap[t])
            plot.add_layout(label)

        if self.report:
            filename = self.logname[:len(self.logname)-4] + "_cluster.html"
            output_file(filename)
            save(plot)
            print("Cluster saved to html")
            self.logger.info("Cluster saved to html")
        if self._display:
            show(plot)

    @staticmethod
    def postal_code_preprocessing(df, col='authorZipCode'):
        """
        Preprocessing of the column postal code to keep only the first 2 digits \n
        `df`: the dataframe containing a columns of postal codes \n
        `col`: the name of the column containing the postal codes \n
        `return` a series containing the postal codes \n
        """
        postal_codes = df[col].fillna(0).astype(str)
        postal_codes = postal_codes.apply(lambda x: '0' + x[:1] if (len(x) == 1 or x[1] == '.') else (
            x if (len(x) <= 2 or x[2] == '.') else (x[:3] if (x[:2] == '97' or x[:3] == '999') else x[:2])))
        return postal_codes

    @staticmethod
    def region_preprocessing(df, col='authorZipCode'):
        """
        Convert the postal codes to regions of France \n\n
        `df`: the dataframe containing a columns of postal codes \n
        `col`: the name of the column containing the postal codes  \n
        `return` the original dataframe with a new column 'region' \n
        """
        postal_codes = TopicModelling.postal_code_preprocessing(df)
        regions = postal_codes.apply(TopicModelling.find_region)
        df['region'] = regions
        return df

    @staticmethod
    def find_region(postal_code):
        """
        Find the region of a postal code \n\n
        `postal_code`: a postal code with 2 digits \n
        `return` the region of the postal code \n
        """
        REGIONS = {
            'Auvergne-Rhône-Alpes': ['01', '03', '07', '15', '26', '38', '42', '43', '63', '69', '73', '74'],
            'Bourgogne-Franche-Comté': ['21', '25', '39', '58', '70', '71', '89', '90'],
            'Bretagne': ['35', '22', '56', '29'],
            'Centre-Val de Loire': ['18', '28', '36', '37', '41', '45'],
            'Corse': ['2A', '2B'],
            'Grand Est': ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88'],
            'Guadeloupe': ['971'],
            'Guyane': ['973'],
            'Hauts-de-France': ['02', '59', '60', '62', '80'],
            'Île-de-France': ['75', '77', '78', '91', '92', '93', '94', '95'],
            'La Réunion': ['974'],
            'Martinique': ['972'],
            'Normandie': ['14', '27', '50', '61', '76'],
            'Nouvelle-Aquitaine': ['16', '17', '19', '23', '24', '33', '40', '47', '64', '79', '86', '87'],
            'Occitanie': ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82'],
            'Pays de la Loire': ['44', '49', '53', '72', '85'],
            'Provence-Alpes-Côte d\'Azur': ['04', '05', '06', '13', '83', '84'],
        }
        for key,value in REGIONS.items():
            if postal_code in value:
                return key

        return 'Undefined'

    def topic_postal_code(self, region=False, percentage=False):
        """
        Analyse the importance of each topic per postal codes or region \n\n
        `region`: to group the postal codes into region or note \n
        `percentage`: to display in percentage or in number of occurence \n
        """
        if 'authorZipCode' not in self.df.columns:
            return

        if region:
            postal_codes = TopicModelling.region_preprocessing(self.df)['region']
        else:
            postal_codes = TopicModelling.postal_code_preprocessing(self.df)

        postal_codes_index = dict.fromkeys(postal_codes)
        postal_codes_unique = list(postal_codes_index.keys())

        if not region:
            postal_codes_unique = [int(float(x)) for x in postal_codes_unique]
            postal_codes_unique.sort()
            postal_codes_unique = [str(x).zfill(2) for x in postal_codes_unique]

        for key in list(postal_codes_index):
            if not region:
                key = str(int(float(key))).zfill(2)
            postal_codes_index[key] = postal_codes_unique.index(key)

        count_topic_postal = np.zeros((len(postal_codes_unique), self.n_topics))

        for i in tqdm(range(len(self.keys))):
            if not region:
                count_topic_postal[postal_codes_index[str(int(float(postal_codes[i]))).zfill(2)]][self.keys[i]] += 1
            else:
                count_topic_postal[postal_codes_index[postal_codes[i]]][self.keys[i]] += 1

        self.df_count_topic_postal = pd.DataFrame(data=count_topic_postal, index=postal_codes_unique)
        self.df_count_topic_postal.columns = ['Topic {}'.format(i) for i in range(self.n_topics)]

        if percentage:
            self.df_count_topic_postal = self.df_count_topic_postal.div(self.df_count_topic_postal.sum(axis=1), axis=0)
            self.df_count_topic_postal.fillna(0)

        fig, ax = plt.subplots(figsize=(14, 10))
        ax = sb.heatmap(self.df_count_topic_postal, cmap="YlGnBu", ax=ax)

        if self.report:
            self.pdf.savefig(fig)

        if self._display:
            plt.show()
        else:
            plt.close(fig)

    def get_topics_words_weigths_counts(self, n_words=50):
        """
        Create a dataframe containing in each row : a word, its topic, the number of occurence inside the topic,
        and its weigths inside the topics \n\n
        `n_words`: the number of words per topic \n
        the dataframe is accessible inside the attribute self.df_topics_words_weigths_counts \n
        """
        self.topics_words_weigths = self.model.components_ / self.model.components_.sum(axis=1)[:, np.newaxis]

        topics_words_weigths_counts = []

        documents_by_topics = {}
        words_counts_by_topic = {}

        for i in range(self.n_topics):
            documents_by_topics[i] = []
            words_counts_by_topic[i] = {}

        for i in range(len(self.documents)):
            documents_by_topics[self.keys[i]].append(self.documents[i])

        for topic, documents in documents_by_topics.items():
            count_vectorizer = CountVectorizer()
            try:
                count_vectorizer.fit(documents)
            except Exception as e:
                print("type error: " + str(e))
                print(traceback.format_exc())
                continue
            words_counts_by_topic[topic] = count_vectorizer.vocabulary_
            del count_vectorizer

        for i in tqdm(range(self.topics_words_weigths.shape[0])):
            topic_words_weigths = self.topics_words_weigths[i]
            top_n_words_index = np.argsort(topic_words_weigths)[::-1][:n_words]

            for j in tqdm(range(len(top_n_words_index))):
                index = top_n_words_index[j]
                temp_word_vector = np.zeros((1, self.document_term_matrix.shape[1]))
                temp_word_vector[:, index] = 1
                the_word = self.count_vectorizer.inverse_transform(temp_word_vector)[0][0]

                if the_word in words_counts_by_topic[i]:
                    topics_words_weigths_counts.append(
                        [i, the_word, topic_words_weigths[index], words_counts_by_topic[i][the_word]])

        self.df_topics_words_weigths_counts = pd.DataFrame(data=topics_words_weigths_counts,
                                                   columns=['topic_id', 'word', 'importance', 'word_count'])

    def display_wordcloud(self, topic_number):
        """
        Display a wordcloud of the words of a topic by weight \n
        `topic_number`: the number of the topic \n
        """
        df_text = self.df_topics_words_weigths_counts
        df_text = df_text.loc[df_text['topic_id'] == topic_number]
        words = df_text['word'].values
        importance = df_text['importance'].values
        number = df_text['word_count']

        # Transformer en nombre de mots a avoir en fonction de l'importance?
        importance = 10000*importance
        importance = importance.astype(int)

        # Avoir la liste de mots qui va passer dans le wordcloud
        words_total = np.copy(words)
        for i in range(number.shape[0]):
            w = words[i]+" "
            words_total[i] = w*importance[i]

        # Transformer en string
        text = ''.join(words_total)

        # Enlever le char ', facultatif
        text.replace("'", "")

        path_current_directory = os.path.dirname(os.path.abspath(__file__))
        mask = np.array((Image.open(os.path.join(path_current_directory, 'cloud_mask2.png'))))

        # Dessiner le nuage de mots
        wordcloud = wc.WordCloud(background_color="white", collocations=False, mask=mask).generate(str(text))

        fig, ax = plt.subplots(figsize=(16, 8), sharey='all', dpi=160)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.set_axis_off()

        plt.title("Word Cloud by weight of Topic " + str(topic_number))

        if self.report:
            self.pdf.savefig(fig)

        if self._display:
            plt.show()
        else:
            plt.close(fig)

    def display_topics_words_weights_counts(self, n_words=15):
        """
        Plot Word Count and Weights of Topic Keywords \n
        """
        df = self.df_topics_words_weigths_counts
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        for i in range(self.n_topics):

            data = df.loc[df.topic_id == i, :][:n_words]

            if data.empty:
                continue

            fig, ax = plt.subplots(figsize=(16, 8), sharey='all', dpi=160)

            ax.bar(x='word', height="word_count", data=data, color=cols[i%len(cols)], width=0.5,
                   alpha=0.3,
                   label='Word Count')
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="importance", data=data, color=cols[i%len(cols)], width=0.2,
                        label='Weights')
            ax.set_ylabel('Word Count', color=cols[i%len(cols)])
            ax.set_title('Topic: ' + str(i), color=cols[i%len(cols)], fontsize=16)
            ax.tick_params(axis='y', left=False)
            ax.set_xticklabels(data['word'], rotation=30, horizontalalignment='right')
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')

            fig.tight_layout(w_pad=2)
            fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)

            if self.report:
                self.pdf.savefig(fig)

            if self._display:
                plt.show()
            else:
                plt.close(fig)

            self.display_wordcloud(i)

    def topic_correlation(self, topic_matrix):
        """
        Analyse the correlation between topic \n \n
        `topic_matrix`: self.topic_matrix \n
        """
        data = pd.DataFrame(data=topic_matrix)
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(16, 8), sharey='all', dpi=160)
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(data.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.columns)

        fig.suptitle('Topic Correlation')

        if self.report:
            self.pdf.savefig(fig)

        if self._display:
            plt.show()
        else:
            plt.close(fig)

    def summary(self, topic_matrix):
        """
        Run the topic modelling analyse of the topic_matrix given in parameters \n\n
        `topic_matrix`: the topic matrix of the model \n
        """
        self.get_keys(topic_matrix)

        self.keys_to_counts()

        print('Computing top words')
        self.logger.info('Computing top words')
        self.get_top_n_words(30, self.keys, self.document_term_matrix, self.count_vectorizer)

        print('Displaying top words')
        self.logger.info('Displaying top words')
        self.display_top_n_word()

        print('Computing topics per postal code by counts')
        self.logger.info('Computing topics per postal code by counts')
        self.topic_postal_code(region=False, percentage=False)

        print('Computing topics per postal code by percentage')
        self.logger.info('Computing topics per postal code by percentage')
        self.topic_postal_code(region=False, percentage=True)

        self.topic_postal_code(region=True, percentage=False)

        self.topic_postal_code(region=True, percentage=True)

        print('Computing words weights per topics')
        self.logger.info('Computing words weigths per topics')
        self.get_topics_words_weigths_counts(n_words=50)

        print('Displaying words weigths per topics')
        self.logger.info('Displaying words weigths per topics')
        self.display_topics_words_weights_counts(n_words=20)

        print('Topic Correlation')
        self.logger.info('Topic Correlation')
        self.topic_correlation(topic_matrix)

        if self.report:
            self.pdf.close()

        if self._cluster:
            print('Dimensional Reduction')
            self.logger.info('Dimensional Reduction')
            tsne_vectors = self.dimensional_reduction_tsne(topic_matrix)

            print('Get Mean Topic Vectors')
            self.logger.info('Get Mean Topic Vectors')
            self.get_mean_topic_vectors(self.keys, tsne_vectors)

            print('Displaying Cluster')
            self.logger.info('Displaying Cluster')
            self.display_cluster(tsne_vectors)

        print('File Analysed !')
        self.logger.info('File Analysed !')

        self.logger.removeHandler(self.logger_file_handler)
        del self.logger, self.logger_file_handler

