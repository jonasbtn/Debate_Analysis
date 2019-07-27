import os
import sys
modulefolder = '\\'.join(os.path.dirname(os.path.realpath(__file__)).split('\\')[:-1])
sys.path.append(modulefolder)

from sklearn.decomposition import LatentDirichletAllocation

from TopicModelling.Topic_Modelling import TopicModelling


class LDA(TopicModelling):
    """
    A class used to run a Latent Dirichlet Allocation (LDA) on a dataset. \n\n
    Do not forget to run 'pip install -r requirements.txt' to avoid any missing packages errors \n\n
    """

    def __init__(self, folder, filename, columns, n_topics, sample_only=True, display=False, report=False,
                 cluster=False):
        """
        Initialize the LDA object as a child of a Topic Modelling object \n\n
        `folder`: the path of the folder containing the data \n
        `filename`: the name of the file to analyse \n
        `columns`: the column to analyse inside the file \n
        `n_topics`: the number of topics to extract \n
        `sample_only`: to analyze only a sample of the dataset in case of lack of computational power (limits to
        10K observations) \n
        `report`: to generate a pdf report contaning all the output and graphs of the topic modelling analysis \n
        NOTE : The reports are located in the folder containing the data inside the folder "grand_debat_report" \n
        `cluster`: to create a cluster of the topics --> Computer Intensive and very long \n
        `inference`: used to analyse the dataset of all users \n
        """
        TopicModelling.__init__(self, folder, filename, columns, n_topics, sample_only, display, report, cluster)

        self.logger.info("RUNNING LDA on " + filename)
        self.logger.info("Columns to be analysed :")
        self.logger.info(columns)

    def model_LDA(self):
        """
        Create the topic matrix according to the LDA model by using the dedicated module of SkLearn \n
        """
        print("Fitting model")
        self.logger.info("Fitting Model")
        self.model = LatentDirichletAllocation(n_components=self.n_topics, learning_method='online',
                                          random_state=0, verbose=0)
        self.topic_matrix = self.model.fit_transform(self.document_term_matrix)
        print('Model LDA fitted')
        self.logger.info("Model LDA fitted")

    def run(self):
        self.model_LDA()
        self.summary(self.topic_matrix)


if __name__ == '__main__':
    folder = "../../data"
    filename = "DEMOCRATIE_ET_CITOYENNETE.csv"
    column = ['title']
    n_topics = 8

    lda = LDA(folder, filename, ['title'], n_topics, sample_only=True, display=False, report=True, cluster=False)

    lda.run()