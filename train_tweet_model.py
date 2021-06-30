from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

class TweetDFTrain:
    """
    this function will prepare parsed data for modeling
    
    Return
    ------
    dataframe
    """
    
    def __init__(self, processedDf, max_n_grams):
        self.processedDf = processedDf
        self.max_n_grams = max_n_grams
        
    def vectorize_data(self)->None:
        """
        this function will vectorize data for training the model

        Return
        ------
        None
        """
        max_n_grams = self.max_n_grams
        data = {str(i+1)+'_gram':'' for i in range(max_n_grams)}
        
        for i in range(max_n_grams):
            vectorizer = CountVectorizer(ngram_range=(1,i+1))
            vectorizer.fit(self.processedDf['cleaned_tweet'].values)
            data[str(i+1)+'_gram'] = vectorizer.transform(self.processedDf['cleaned_tweet'].values)
            
        self.data = data
        
    def tf_id_data(self)->None:
        """
        this function will compute the tf-id of data for training the model

        Return
        ------
        None
        """
        data_1 = self.data
        data = {}
        vectorized_data_tf_id = {}
        for key in data_1.keys():
            transformer = TfidfTransformer()
            transformer.fit(data_1[key])
            data[key+'_tf_id'] = transformer.transform(data_1[key])
        data_1.update(data)  
        self.data = data_1
        
    def train_model(self)->dict:
        """
        this function will train and select the optimized model for each of the parsed proceeded variant of data

        Return
        ------
        dictionary
        """
        data = self.data
        y = self.processedDf['label']
        models = {}
        for key in data.keys():
            X=data[key]
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75, stratify=y)
            
            model = SGDClassifier()
            distributions = dict(
                loss=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                learning_rate=['optimal', 'invscaling', 'adaptive'],
                eta0=uniform(loc=1e-7, scale=1e-2),
                penalty=['l1', 'l2', 'elasticnet'],
                alpha=uniform(loc=1e-6, scale=1e-4)
            )
            random_search_cv = RandomizedSearchCV(
                estimator=model,
                param_distributions=distributions,
                cv=5,
                n_iter=50
            )
            random_search_cv.fit(X_train, y_train)
            model = random_search_cv.best_estimator_
            model.fit(X_train, y_train)
            model_data = {'X_train':X_train, 'X_valid':X_valid, 'y_train':y_train, 'y_valid':y_valid, 'model':model}
            models[key+'_model'] = model_data
        return models   

def get_trained_models(df:pd.DataFrame,max_n_grams:int)->dict:
    """
    this function will vectorize, compute tf-id and train the models. It takes the processed data as a first argument 
    and maximum n-grams as a secon argument

    Return
    ------
    dictionary
    """
    models = TweetDFTrain(df,max_n_grams)
    models.vectorize_data()
    models.tf_id_data()
    trained_models = models.train_model()
    return trained_models