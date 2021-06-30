import warnings
warnings.filterwarnings('ignore')     # This prevents displays of warnings which can be a distruction to viewing outputs
import pickle

#imports from local modules
from modules.extract_dataframe import read_json, TweetDfExtractor
from modules.preparation_tweets import get_df_info, get_cleaned_tweet_and_data_for_model
from modules.topic_modeling import TweetDFTopicModeling
from modules.visualization_tweets import TweetDfVisualization
from modules.train_tweet_model import get_trained_models
from modules.model_evaluate import TweetDFModelEvaluate

data_source = "./data/covid19.json"

# reading the data and putting the total number of entries (tweet_len) and data (tweet_list) in variables
tweet_len, tweet_list = read_json(data_source)

# creates an instance of TweetDfExtractor
tweet = TweetDfExtractor(tweet_list)
# creates pandas dataframe using get_tweet_df method of TweetDfExtractor    
tweet_df = tweet.get_tweet_df()         

# display information from the tweet data
get_df_info(tweet_df)
# get the cleaned data for visualization and model building
cleaned_tweet, data_for_model = get_cleaned_tweet_and_data_for_model(tweet_df)

# create an imstance of TweetDFtopicModeling class
model_topic = TweetDFTopicModeling(data_for_model)
# prepare data for topic modeling
model_topic.prepare_data()
# create the topic modeling model
lda = model_topic.creat_topic_model()
# create the visualization of the model topics
lda_viz = model_topic.viz_lda_topics(lda_model=lda)
# display the visualized topics from the topic modeling model
lda_viz

# create an instance of the TweetDfVisualization class
viz = TweetDfVisualization(cleaned_tweet)
# create a wordcloud of most mentioned words
viz.create_wordcloud()
# create charts of the polarity and subjectivity of tweets
viz.create_viz()

# get a group of trained classifiers
models = get_trained_models(data_for_model, 3)

# create an instance of TweetDFModelEvaluate
classifier_models = TweetDFModelEvaluate(models)
# evaluate the models
classifier_models.evaluate_model()
# select the best performing model with its meta data
classifier_model_dict = classifier_models.select_model()
# get the selected model
classifier_model = classifier_model_dict['model']
# get the model name
classifier_model_name = classifier_model_dict['name']
# set the destination folder
destination_folder = 'models'
# create the name with which the model will be saved
filename = classifier_model_name + '.sav'
# create the address to save 
address = destination_folder + '/' + filename
# save the model
pickle.dump(classifier_model, open(address, 'wb'))