import pandas as pd
from collections import Counter
import lyrics as ly

def csv_parser(filename):
    """
    Reads the file and performs all the Data Pre-Processing steps
    Args:
        filename (str): name of the file
    Returns:
        results (dict): dictionary that contains the wordcount
                        and a list of cleaned words
    """

    lyr = ly.Lyrics()

    # reading the file into a pandas dataframe
    df = pd.read_csv(filename)

    # dropping all null values and columns that are not required
    df = df.drop(columns=["Unnamed: 0", "Title", "Album", "Year", "Date"])
    df = df.dropna()

    # removing leading/trailing spaces & punctuations
    df = df.replace(r'''[^A-Za-z' ]''', ' ', regex=True)

    # converting to lowercase
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

    # making a list of strings for each lyric (row)
    df['Lyric'] = df['Lyric'].str.split()

    # stop_words = set(stopwords.words('english'))

    # creating an empty list to store all words that occur in the lyrics column
    all_words = []

    # looping through every row in the lyrics column
    for lyric in df["Lyric"]:
        # adding the list of words to the empty list
        all_words += lyric

    # creating an empty list to store words excluding stopwords
    cleaned_words = []

    # looping through every word in the all_words list
    for i in all_words:
        # checking if the word is in the stopwords file
        if i not in lyr.load_stop_words("stopwords.txt"):
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
