import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # need this to iterate
    xLenghts = test_set.get_all_Xlengths()


    for X, lenghts in xLenghts.values():
        # Initiate them here as need to be reset each iteration
        bestScore = float("-inf")
        # To add to guesses (List of best words)
        bestGuess = None
        # Need to keep the logls for each word in a dict
        logLike = {}
        # Take the pre selected model and get words so log like can be calced
        for word, model in models.items():
            try:
                score = model.score(X, lenghts)
                logLike[word] = score

                if score > bestScore:
                    bestScore = score
                    bestGuess = word
            # If cannot process word return None
            # TODO: check if None is the correct thing to return
            except:
                logLike[word] = float("-inf")

        # Append the best guess for this iteration
        guesses.append(bestGuess)
        # Append the dictionary for log likelihood for each word
        probabilities.append(logLike)


    return probabilities, guesses
