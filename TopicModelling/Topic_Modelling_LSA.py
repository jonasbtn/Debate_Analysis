import os
import sys
modulefolder = '\\'.join(os.path.dirname(os.path.realpath(__file__)).split('\\')[:-1])
sys.path.append(modulefolder)

from sklearn.decomposition import TruncatedSVD
from TopicModelling.Topic_Modelling import TopicModelling


class LSA(TopicModelling):
    """
    A class used to run a Latent Semantic Analysis (LSA) on a dataset. \n\n
    Do not forget to run 'pip install -r requirements.txt' to avoid any missing packages errors \n\n
    """
    def __init__(self, folder, filename, columns, n_topics, sample_only=True, display=False, report=False,
                 cluster=False):
        """
        Initialize the LSA object as a child of a Topic Modelling object \n\n
        `folder`: the path of the folder containing the data  \n
        `filename`: the name of the file to analyse \n
        `columns`: the column to analyse inside the file \n
        `n_topics`: the number of topics to extract \n
        `sample_only`: to analyze only a sample of the dataset in case of lack of computational power (limits to
        10K observations) \n
        `display`: to display each plot in a new windows \n
        `report`: to generate a pdf report contaning all the output and graphs of the topic modelling analysis \n
        NOTE : The reports are located in the folder containing the data inside the folder "grand_debat_report" \n
        `cluster`: to create a cluster of the topics --> Computer Intensive and very long \n
        `inference`: used to analyse the dataset of all users \n
        """
        TopicModelling.__init__(self, folder, filename, columns, n_topics, sample_only, display, report, cluster)

    def model_LSA(self):
        """
        Create the topic matrix according to the LSA model : a trucated SVD to keep only the n most important topics \n
        """
        print("fitting model")
        self.logger.info("Fitting model")
        self.model = TruncatedSVD(n_components=self.n_topics)
        self.topic_matrix = self.model.fit_transform(self.document_term_matrix)
        print('Model LSA fitted')
        self.logger.info("Model LSA fitted")

    def run(self):
        self.model_LSA()
        self.summary(self.topic_matrix)


if __name__ == '__main__':
    folder = "../../data"
    filename = "DEMOCRATIE_ET_CITOYENNETE.csv"
    column = ['title']
    n_topics = 8

    lsa = LSA(folder, filename, column, n_topics, sample_only=True, display=False)

    lsa.run()






