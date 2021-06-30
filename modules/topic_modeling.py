from gensim import corpora
from gensim.models import ldamodel, CoherenceModel
from gensim.parsing.preprocessing import remove_stopwords
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

class TweetDFTopicModeling:
    """
    this function will create topic modeling model parsed data for modeling
    
    Return
    ------
    dict
    """
    
    def __init__(self, df):
        self.data = df
        
    def prepare_data(self)->None:
        """
        this function will prepare data for creating topic modeling model 
        Return
        ------
        None
        """
        
        data = self.data
        #Converting tweets to list of words For feature engineering
        sentence_list = [remove_stopwords(tweet) for tweet in data['cleaned_tweet']]
        word_list = [sent.split() for sent in sentence_list]

        #Create dictionary which contains Id and word 
        word_to_id = corpora.Dictionary(word_list)
        corpus_1= [word_to_id.doc2bow(tweet) for tweet in word_list]
        data = {'word_list':word_list, 'word_to_id':word_to_id, 'corpus':corpus_1 }
        
        self.data = data
        
    def creat_topic_model (self)->dict:
        """
        this function will create topic modeling model 

        Return
        ------
        dict
        """
        data = self.data
        corpus = data['corpus']
        word_to_id = data['word_to_id']
        lda_model = ldamodel.LdaModel(corpus,
                                           id2word=word_to_id,
                                           num_topics=3, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=50,
                                           passes=50,
                                           alpha='auto',
                                           per_word_topics=True)
        self.model = lda_model
        return lda_model
    
    def viz_lda_topics(self):
        
        data = self.data
        lda_model = self.model
        corpus = data['corpus']
        word_to_id = data['word_to_id']
        # Visualize the topics
        pyLDAvis.enable_notebook()

        LDAvis_prepared = gensimvis.prepare(lda_model, corpus, word_to_id)
        return LDAvis_prepared