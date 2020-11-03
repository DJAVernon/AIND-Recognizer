import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # USED THIS for inspiration: https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
        # Specifically for the BIC formulas - Where are we meant to find this?
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Again same format as the other two to iterate and to keep track
        # This time need to minimise score not maximise...
        bestScore = float("inf")
        bestModel = None
        for states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(states)
                log_like = model.score(self.X, self.lengths)
                np.seterr(divide='ignore')
                logN = np.log(len(self.X))
                # Instead of the usual ** i take 1 from states as it seems to yeild better results
                # P seems to have various formulaes - this worked best
                p = states * (states-1) + 2 * len(self.X[0]) * states

                # Ignores the divide by 0 zero (caused by floats?)

                bic = -2 * log_like + p * logN
                # HAS TO BE THE MINIMUM VALUE NOT MAX
                # Results not performing as expected - why?

                if bic < bestScore:
                    bestScore = bic
                    bestModel = model

            except Exception:
                pass
            except ZeroDivisionError:
                pass
            except RuntimeError:
                pass

        return bestModel if bestModel is not None else self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # As before set bestscore as -inf
        bestScore = float("-inf")
        bestModel = None
        # Create the list of vocab for this selector
        vocab = self.words
        for states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(states)
                log_like = model.score(self.X,self.lengths)
                score = 0

                # Iterate through and remove relevant word - More efficent way to do this?
                # Yes moved this to inside this for loop so don't have to go through twice
                # and dont have to move to another list - Much better -DELETE THIS BEFORE SUBMISSION
                for word in vocab:
                    if word == self.this_word:
                        pass
                    else:
                        X, lengths = self.hwords[word]
                        score = score + model.score(X,lengths)

                # Use the formulae provided to get the score
                calculation = (log_like - score) / (len(self.words) - 1)
                # Compare it with the current best
                if calculation > bestScore:
                    bestScore = calculation
                    bestModel = model

            except Exception:
                pass

            return bestModel if bestModel is not None else self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    #select best model based on average log Likelihood of cross-validation folds
    # no source for help for this one luckily have done it at uni before..


    def select(self):
        # To remove deprecated - Already here in code
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        split = KFold(n_splits = 3)
        bestScore = float("-inf")
        best_model = None
        logList = []

        # Iterate through each model
        for states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(states)
                # Must do folding for cross validation - Should i do k fold or standard?
                # TODO: Decide if k fold or standard is better - test both?

                # Must be greater than 2 to use cross validation
                if len(self.sequences) > 2:
                    # Split the sequences up into training and test
                    for training, test in split.split(self.sequences):

                        # Combine data so it is usable..
                        self.X, self.lengths = combine_sequences(training,self.sequences)
                        # same for test sequence
                        x_test, lengths_test = combine_sequences(test,self.sequences)
                        # Compute log
                        log_like = model.score(x_test,lengths_test)
                else:
                    log_like = model.score(self.X,self.lengths)

                logList.append(log_like)

                average = np.mean(logList)
                if average > bestScore:
                    bestScore = average
                    best_model = model

            except Exception:
                pass

            # If fails then just return best model is consstant
            return best_model if best_model is not None else self.base_model(self.n_constant)



