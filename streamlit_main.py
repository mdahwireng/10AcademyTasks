import streamlit as st
from modules.extract_dataframe import read_json, TweetDfExtractor
from modules.preparation_tweets import get_df_info, get_cleaned_tweet_and_data_for_model
from modules.visualization_tweets import TweetDfVisualization
from modules.topic_modeling import TweetDFTopicModeling
from streamlit import components

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

dataset_text = "Data was downloaded from twitter and saved as a json file. The data was downloaded using ‘COVID19 Africa’, ‘COVID19 Vaccination Africa’ and ‘Sars-Cov2 Mutation Africa’ as keywords. The data has 6532 entries with 15 columns. There were two columns with missing values. The total number of missing values across the 2 columns were 7458. The columns with the missing values were ‘possibly-sensitive’ and ‘place’. The complete list of columns was 'created_at', 'source', 'original_text', 'polarity', 'subjectivity','lang', 'favorite_count', 'retweet_count', 'original_author', 'followers_count', 'friends_count', 'possibly_sensitive', 'hashtags', 'user_mentions',and  'place'."
intro = "WHO wants to be aware of the level of awareness of covid-19 and seriousness attached to covid19 related issues across the globe. They want to know when there is a drop in the observation and discussion of covid-19 protocols and related issues. They want to notice when there is a panic originating from the fear of covid-19. Also, WHO will want to monitor the readiness of people to take covid-19 vaccines on a regular basis to be able to detect the general view of vaccination on the continent. These data will help them make informed decisions, plan policies and tailor action plan in response to changes in human behavior towards covid-19 and its related issues. These objectives will be achieved through Natural Language Processing techniques, Topic Modelling and Sentiment analysis to be precise. These techniques will bring out the hidden topics in the discussions surrounding covid-19 and give a clear indication of whether the world is positive or speaking ill of covid-19 related issues as well as pointing out if the discussions are opinionated or more drawn towards the side of facts. To capture the views of people across the globe, it is important to collect data from a point where the world meet and there is little or no fear of expressing one’s thoughts and views. For these reasons twitter was chosen as the place to mine data. To ensure the sustainability, scalability, and modularization of the processes an MLOps pipeline was set up. DATA"
methods_text = "The sentiments score was acquired through Textblob. The data was cleaned by removing duplicates, emojis, punctuation marks and digits from them. Emoji, Re were some of the packages used for the cleaning of data. Visualization of data was done with Seaborn and Matplotlib. The MLOps pipeline is made up of 16 components namely Data Extraction, Visualization and Report,Data Validation, Model Evaluation, Data Preparation, Model Training, Source Code Repository, Model Registry, Deployment, Prediction, Performance  Monitoring, Triger, Continuous Integration, Database, Data Minig and Exploration Data Analysis (EDA )."

header = st.beta_container()
dataset = st.beta_container()
method = st.beta_container()
plots = st.beta_container()

loc = st.sidebar.radio('NAVIGATION', ['About', 'Insights', 'Model vizualization'])

if loc == 'About':

    with header:
        st.title('This Is A Tweet classification Project')
        st.header('Introduction')
        st.write(intro)


    with dataset:
        st.header('Dataset')
        st.write(dataset_text)



    with method:
        st.header('Methhods')
        st.write(methods_text)

if loc == 'Insights':
    st.title('This Is A Tweet classification Project')
    # create an instance of the TweetDfVisualization class
    viz = TweetDfVisualization(cleaned_tweet)
    # create a wordcloud of most mentioned words
    wordcloud_fig = viz.create_wordcloud(output=True)
    # create charts of the polarity and subjectivity of tweets
    chart_collection_fig = viz.create_viz(output=True)

    with plots:
        st.header('Insights From The Dataset')
        st.subheader('Charts Loading...')
        st.pyplot(wordcloud_fig)
        st.pyplot(chart_collection_fig)

if loc == 'Model vizualization':
    st.header('LDA Model Vizsualization')
    st.subheader('Model Loading...')
    # create an imstance of TweetDFtopicModeling class
    model_topic = TweetDFTopicModeling(data_for_model)
    # prepare data for topic modeling
    model_topic.prepare_data()
    # create the topic modeling model
    lda = model_topic.creat_topic_model()
    # create the visualization of the model topics
    lda_viz = model_topic.viz_lda_topics(streamlit=True, lda_model=lda)
    # display the visualized topics from the topic modeling model
    components.v1.html(lda_viz, width=1000, height=800, scrolling=True)
