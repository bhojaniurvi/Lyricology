"""
DS3500: Advanced Programming with Data
Homework 3
Rishita Shroff, Urvi Bhojani, Kush Ramchand Raimalani

Reusable Framework for Natural Language Processing (NLP)
Building a reusable framework, that can handle a variety of datasets,
for comparative text analysis.
"""

# importing libraries
import pandas as pd
from collections import Counter, defaultdict
import sankey as sk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import custom_parser as cp


class NLPException(Exception):
    """
    A user-defined exception class for signalling a NLP Framework
    specific issue
    """
    def __init__(self, artist, msg=""):

        # initializing parent class and parameters
        super().__init__("Can't Process Artist Lyrics")
        self.artist = artist
        self.msg = msg


# creating class object
class Lyrics:
    """
    An instance or class consisting of song lyrics of different
    artists where:
    self.data = default dictionary
    _default_parser() helper method to pre-process data of a file
    _save_results() parsers results into internal state
    load_text() processes an Artist's csv file
    load_stop_words() processes the stopwords text file
    wordcount_sankey() generates a dataframe for the most common words
                       required to create a sankey diagram
    plot_wordcloud() generates subplots of word clouds for each artist
    plot_sentiment() generates a bar chart to track the sentiment score
                     for each artist
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the Lyrics object
        """
        self.data = defaultdict(dict)

    @staticmethod
    def _default_parser(filename):
        """
        Reads a file and performs all the Data Pre-Processing steps

        Args:
            filename (str): name of the file

        Returns:
            results (dict): dictionary that contains the wordcount
                            and a list of cleaned words
        """

        with open(filename, "r", encoding="utf8") as file:
            # reading every line in the file
            data = file.read()

            # converting the file to a list
            allwords_lst = data.replace("\n", " ").split(" ")

            # creating an empty list to store words excluding stopwords
            cleaned_words = []

            # looping through every word in the all_words list
            for i in allwords_lst:
                # checking if the word is in the stopwords file
                if i not in Lyrics.load_stop_words("stopwords.txt"):
                    # adding the word to the empty list
                    cleaned_words.append(i)

            # getting the total number of times a word has been used by each artist
            word_count = dict(Counter(cleaned_words))

            # creating a dictionary that contains wordcount (for sankey & wordcloud)
            # and clean_text (for sentiment analysis)
            results = {"wordcount": word_count,
                       "clean_text": " ".join(cleaned_words)}

        # returning results
        return results

    def _save_results(self, label, results):
        """ Integrate parsing results into internal state

        Args:
            label (str): unique label for a text file that we parsed
            results (dict): the data extracted from the file as a
                            dictionary attribute-->raw data

        Returns:
            None
        """
        for k, v in results.items():
            self.data[k][label] = v

    def load_text(self, filename, label=None, parser=None):
        """ Register a document with the framework

        Args:
            filename (str): name of the file
            label (str): unique name for each file
            parser (staticmethod): function for data
                    pre-processing if not the same file type
                    or default

        Returns:
            None
        """
        try:
            # do default parsing of standard .csv file
            if parser is None:
                results = self._default_parser(filename)
            else:
                results = parser(filename)

            if label is None:
                label = filename

            self._save_results(label, results)

        except Exception as e:
            # raising a NLPException
            print(str(e))
            raise NLPException(filename, str(e))

    @staticmethod
    def load_stop_words(stopfile):
        """
        Reads a file that contains stop words such as
        "a", "an", "the"

        Args:
            stopfile (str): name of file

        Returns:
            stopwords_list (list): list of stopwords
        """

        # reads the stop words file (each line is one word)
        with open(stopfile, "r", encoding="utf8") as file:
            # reading every line in the file
            data = file.read()

            # converting the file to a list
            stopwords_lst = data.replace("\n", " ").split(" ")

        # returning a list of stop words
        return stopwords_lst

    def wordcount_sankey(self, k=5):
        """
        Creating a dataframe of the word counts for all files (artists)
        and sorting it to find the k most common words for each artist

        Args:
            k (int): the number of most common words to be found

        Returns:
            main_sankey_df (dataframe): a concatenated dataframe
                            of all artists and their corresponding
                            words and counts
        """

        # extracting the value of key = wordcount
        results = self.data['wordcount']

        # creating a list to store multiple dataframes
        main_sankey_list = []

        # looping through the dictionary for every artist
        for key, value in results.items():

            # creating a dataframe of the word and its corresponding count
            sankey_df = pd.DataFrame(value.items(), columns=['Word', 'Count'])

            # adding a new column (Artist) to the dataframe
            sankey_df['Artist'] = [key]*sankey_df.shape[0]

            # sorting the dataframe in descending order by count
            sankey_df = sankey_df.sort_values(by='Count', ascending=False)

            # finding the k most common words
            sankey_df = sankey_df.iloc[:k, :]

            # adding the dataframe to the list
            main_sankey_list.append(sankey_df)

        # joining all the dataframes in the list together
        main_sankey_df = pd.concat(main_sankey_list)

        return main_sankey_df

    def plot_wordcloud(self, row=3, col=3):
        """
        Creating WordClouds for each Artist to show the
        words that have been sung by every artist

        Args:
            row (int): number of rows the subplots should
                       be arranged in
            col (int): number of columns the subplots should
                       be arranged in

        Returns:
            None
        """

        # extracting the value of key = wordcount
        results = self.data["wordcount"]

        # creating WordCloud features
        wc = WordCloud(background_color="#F2F2F2", max_font_size=90, random_state=42,
                       width=500, height=300)

        plt.figure(figsize=(30, 25))

        # adding a title to the visualization
        plt.suptitle("Artist WordClouds", fontsize=75)

        # looping through every dictionary
        for index, value in enumerate(results):

            # creating a list that contains the word count
            top_dict = list(results.values())[index]

            # generating the wordcloud
            wc.generate_from_frequencies(top_dict)

            # creating a subplot
            plt.subplot(row, col, index + 1)

            # enhancing the wordcloud
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")

            # creating a title for each subplot
            plt.title(f"{value}", fontsize=30)

        # spacing the subplots
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

    def plot_sentiment(self):
        """
        Creating a comparative bar chart on the sentiment analysis
        of each artist (file)
        """

        # extracting the value of key = clean_text
        results = self.data["clean_text"]

        # creating an empty list to store artist names
        # and the polarity/sentiment scores
        artist = []
        polarity_scores = []

        # looping through the cleaned text dictionary
        for key, value in results.items():

            # adding the artist name (key) to artist list
            artist.append(key)

            # finding the polarity scores using TextBlob Library
            # scores returned between [-1, 1] where
            # -1 = negative, 1 = positive
            polarity = TextBlob(value).sentiment.polarity

            # making a list of all polarity scores
            polarity_scores.append(polarity)

        # creating a list of colors
        color_list = []

        # looping through the scores
        # pos=green, neg=red, neutral=blue
        for i in polarity_scores:
            if i > 0:
                color = "limegreen"
                color_list.append(color)
            elif i < 0:
                color = "tab:red"
                color_list.append(color)
            elif i == 0:
                color = "blue"
                color_list.append(color)

        # creating a bar graph for the polarity/sentiment scores of each artist
        plt.bar(x=artist, height=polarity_scores, color=color_list)

        # adding labels to each bar
        for i in range(len(artist)):
            plt.text(i, round(polarity_scores[i], 2), round(polarity_scores[i], 2), ha="center",
                     fontsize=7)

        # enhancing the bar graph
        plt.xticks(rotation=90, fontsize=5)
        plt.yticks(rotation=0, fontsize=5)

        # adding title and axis labels
        plt.title("Sentiment Scores for Each Artist")
        plt.xlabel("Artist Name")
        plt.ylabel("Sentiment Score")
        plt.show()

def main():

    # calling the class
    lyrics = Lyrics()

    # loading 9 files
    lyrics.load_text("TaylorSwift.csv", label='Taylor Swift', parser=cp.csv_parser)
    lyrics.load_text("EdSheeran.csv", label='Ed Sheeran', parser=cp.csv_parser)
    lyrics.load_text("Eminem.csv", label='Eminem', parser=cp.csv_parser)
    lyrics.load_text("NickiMinaj.csv", label='Nicki Minaj', parser=cp.csv_parser)
    lyrics.load_text("JustinBieber.csv", label='Justin Bieber', parser=cp.csv_parser)
    lyrics.load_text("DuaLipa.csv", label='Dua Lipa', parser=cp.csv_parser)
    lyrics.load_text("Khalid.csv", label='Khalid', parser=cp.csv_parser)
    lyrics.load_text("Drake.csv", label='Drake', parser=cp.csv_parser)
    lyrics.load_text("BillieEilish.csv", label='Billie Eilish', parser=cp.csv_parser)

    # Part 1: Sankey Diagram
    # calling the wordcount_sankey function and making
    df = lyrics.wordcount_sankey()
    sk.make_sankey(df, 'Artist', 'Word', 'Count')

    # Part 2: Visualization with Subplots (WordClouds)
    lyrics.plot_wordcloud()

    # Part 3: Comparative Visualization (Bar Chart)
    lyrics.plot_sentiment()

    # Testing the Exception
    # lyrics.load_text("TaylorSwift.csv", label='Taylor Swift', parser=cp.csv_parser)

if __name__ == "__main__":
    main()
