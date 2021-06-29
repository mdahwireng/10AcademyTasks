import pandas as pd
import emoji
import string
import re

class TweetDfDataPreparation:
    """
    this function will prepare tweets data form tweet dataframe for modelling and visualization
    
    Return
    ------
    dataframe
    """
    def __init__(self, tweets_df):
        
        self.tweets_df = tweets_df
        
    def print_df_info(self) -> None:
        """
        this function will print the info of the tweets datafame

        Return
        ------
        None
        """
        #save the number of columns and names
        col_info = 'The number of colum(s): {}.\nThe column(s) is/are : {} and {}'.format(len(self.tweets_df.columns),','.join(self.tweets_df.columns[:-2]), self.tweets_df.columns[-1])  
        
        #save the number of rows
        num_rows = "\nThe total number of rows: {}".format(len(self.tweets_df))
        
        #save the number of duplicate tweets
        num_dup_tweets = '\nThe number of duplicate tweets: {}'.format(len(self.tweets_df)-len(self.tweets_df.original_text.unique()))
        
        na_cols = self.tweets_df.columns[self.tweets_df.isnull().any()]
        
        #save the number of missing values
        num_na_cols = "\nThe number of columns having missing value(s): {}".format(len(na_cols))
        
        #save the columns with missing value and the num of values missing
        na_cols_num_na = ''
        
        for col in na_cols:
            na_cols_num_na += "\nThe number of rows with missing value(s) in [{}]: {}".format(col, self.tweets_df[col].isnull().sum())
        
        # save the total number of missing values
        tot_na = "\nThe total number of missing value(s): {}".format(self.tweets_df.isnull().sum().sum())
        
        print(col_info, num_rows, num_dup_tweets, num_na_cols, na_cols_num_na, tot_na)
        
        
    def slice_dataframe(self, columns=['created_at', 'original_text', 'polarity', 'subjectivity'],output=True)->pd.DataFrame:
        """
        this function will slice of the tweets datafame. it takes a list of columns to slice and a bolean, output. 
        If its True it returns cleaned tweet

        Return
        ------
        dataframe if output=True, None if output=False
        """
        #sliced_tweet_df = self.tweets_df[columns]
        self.sliced_tweet_df = self.tweets_df[columns]
        if output:
            return self.sliced_tweet_df
        return None
    
    def drop_tweet_dup(self, column_name='original_text',output=True)->pd.DataFrame:
        """
        this function will drop duplicates tweets in slicedtweet datafame. 
        it takes the name of column with the tweets in string format as an argument and 
        a bolean, output. If its True it returns cleaned tweet

        Return
        ------
        dataframe if output=True, None if output=False
        """
        sliced_tweet_df = self.sliced_tweet_df
        sliced_tweet_df.drop_duplicates([column_name], inplace=True)
        self.sliced_tweet_df = sliced_tweet_df
        
        if output:
            return self.sliced_tweet_df
        return None
        
        
    def clean_tweet(self, column_name='original_text', cleaned_tweet_column_name='cleaned_tweet', output=True)->pd.DataFrame:
        """
        this function will clean tweets in slicedtweet datafame. 
        it takes the name of column with the tweets and that of the new column for the cleaned tweet 
        both in string format as an argument and a bolean, output. If its True it returns cleaned tweet

        Return
        ------
        dataframe if output=True, None if output=False
        """
        unwanted = ["'\n',''"]
        take_out = ''
        for char in unwanted:
            take_out = string.punctuation + char 
        
        def remove_punct_and_clean(tweet)->str:
            # removes emojis
            tweet = emoji.get_emoji_regexp().sub(r'', tweet)
            # removes punctuations and newline characters
            tweet  = "".join([char for char in tweet if char not in take_out])
            # removes digits
            tweet = re.sub('[0-9]+', '', tweet)
            # converts to lowercase
            tweet = tweet.lower()
            return tweet
        
        sliced_tweet_df = self.sliced_tweet_df
        sliced_tweet_df[cleaned_tweet_column_name] = sliced_tweet_df[column_name].apply(remove_punct_and_clean)
        self.sliced_tweet_df = sliced_tweet_df
        if output:
            return self.sliced_tweet_df
        return None
        
    def convert_to_datetime(self, column_name='created_at', output=True)->pd.DataFrame:
        """
        this function will convert a parsed column in sliced tweet datafame to datetime. 
        it takes the name of column with the dates in string format as an argument and a bolean, output. 
        If its True it returns dataframe with the converted dates

        Return
        ------
        dataframe if output=True, None if output=False
        """
        sliced_tweet_df = self.sliced_tweet_df
        sliced_tweet_df[column_name] = pd.to_datetime(sliced_tweet_df[column_name])
        self.sliced_tweet_df = sliced_tweet_df

        if output:
            return self.sliced_tweet_df
        return None
    
    def classify_polarity(self, column_name='polarity', cassified_column_name='classified_polarity', output=True)->pd.DataFrame:
        """
        this function will classify a pared column in sliced tweet datafame with polarity. 
        it takes the name of column with the polarity and that of the new column for the classified polarityscores 
        both in string format as an argument and a bolean, output. If its True it returns a dataframe with the classified column added

        Return
        ------
        dataframe if output=True, None if output=False
        """
        
        def classify(value)->str:
            """
            this function will classify numbers. it takes the number be calssified as an argument
            
            Return
            --------
            string
            """
            if value > 0.5:
                return 'very positive'
            elif value > 0.05:
                return 'positive'
            elif value < -0.5:
                return 'very negative'
            elif value < -0.05:
                return 'negative'
            else:
                return 'neutral'
            
        sliced_tweet_df = self.sliced_tweet_df
        sliced_tweet_df[cassified_column_name] = sliced_tweet_df[column_name].apply(classify)
        self.sliced_tweet_df = sliced_tweet_df
        if output:
            return self.sliced_tweet_df
        return None
        
        
    def classify_subjectivity(self, column_name='subjectivity', cassified_column_name='classified_subjectivity', output=True)->pd.DataFrame:
        """
        this function will classify a parsed column in sliced tweet datafame with subjectivity. 
        it takes the name of column with the polarity and that of the new column for the classified subjectivity scores 
        both in string format as an argument and a bolean, output. If its True it returns a dataframe with the classified column added

        Return
        ------
        dataframe if output=True, None if output=False
        """

        def classify(value)->str:
            """
            this function will classify numbers. it takes the number be calssified as an argument

            Return
            --------
            string
            """
            if value > 0.7:
                return 'very subjective'
            elif value > 0.2:
                return 'subjective'
            else:
                return 'factual'

        sliced_tweet_df = self.sliced_tweet_df
        sliced_tweet_df[cassified_column_name] = sliced_tweet_df[column_name].apply(classify)
        self.sliced_tweet_df = sliced_tweet_df
        if output:
            return self.sliced_tweet_df
        return None
    
    def create_data_for_model(self, column_list=['cleaned_tweet','polarity'], polarity_column='polarity', output=True)->pd.DataFrame:
        
        def classify(value)->int:
            """
            this function will classify numbers. it takes the number be calssified as an argument

            Return
            --------
            string
            """
            if value > 0:
                return 1
            else:
                return 0
        x = self.sliced_tweet_df[column_list]
        x = x[x[polarity_column] != 0].reset_index(drop=True)
        x['label'] = x[polarity_column].apply(classify)
        columns = list(x.columns)
        columns.remove(polarity_column)
        data_for_model = x[columns]
        return data_for_model


def get_cleaned_tweet_and_data_for_model(uncleaned_tweet_df)->pd.DataFrame:
    prep = TweetDfDataPreparation(uncleaned_tweet_df)
    prep.slice_dataframe(output=False)
    prep.drop_tweet_dup(output=False)
    prep.convert_to_datetime(output=False)
    prep.classify_polarity(output=False)
    prep.classify_subjectivity(output=False)
    cleaned = prep.clean_tweet()
    data_for_model = prep.create_data_for_model()
    return cleaned, data_for_model

def get_df_info(df)->None:
    prep = TweetDfDataPreparation(df)
    prep.print_df_info()