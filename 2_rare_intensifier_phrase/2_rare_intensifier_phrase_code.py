import nltk
import random
import pickle
import os
from nltk.corpus import *
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import FreqDist

def get_restrictive_intensifier():
    # 1. Make set of noun and adjective which has intensity concept separately from reference corpus (Gutenberg)
    noun_set = []
    adj_set = []
    intensity_keyword = ['degree', 'intensity']

    # Check whether data is produced already or not. Load the data if data is produced.
    if os.path.isfile('noun_set.txt') and os.path.isfile('adj_set.txt'):
        with open('noun_set.txt', 'rb') as f:
            noun_set = pickle.load(f)
        with open('adj_set.txt', 'rb') as f:
            adj_set = pickle.load(f)
        print("noun_set and adj_set data exists. Loaded data\n")

    # If data is not produced yet, start the process. Make noun_set and adj_set
    else:
        total_token = []

        # Get tokenized words from gutenberg corpus
        for each_fileid in gutenberg.fileids():
            each_tokenized_text = set(gutenberg.words(each_fileid)) # eliminate duplicated words
            for each_token in each_tokenized_text:
                total_token.append(each_token.lower()) # make all characters to lower case

        total_token = set(total_token)
        print('total words number for reference is %d' % len(total_token)) # for debugging

        total_token_with_tags = pos_tag(total_token) # tag part-of-speech to words
        for each_word in total_token_with_tags:
            # Select nouns among words
            if each_word[0].isalpha() and each_word[1] == "NN":
                found_keyword = False

                for synset in wn.synsets(each_word[0]):
                    for each_keyword in intensity_keyword:
                        if each_keyword in synset.definition():
                            found_keyword = True
                            break

                    if found_keyword:
                        break

                if found_keyword:
                    noun_set.append(each_word[0])

            # Select adjectives among words
            if each_word[0].isalpha() and each_word[1] == "JJ":
                found_keyword = False

                for synset in wn.synsets(each_word[0]):
                    for each_keyword in intensity_keyword:
                        if each_keyword in synset.definition():
                            found_keyword = True
                            break

                    if found_keyword:
                        break

                if found_keyword:
                    adj_set.append(each_word[0])

        # Save the produced data into file
        with open('noun_set.txt', 'wb') as f:
            pickle.dump(noun_set, f)
        with open('adj_set.txt', 'wb') as f:
            pickle.dump(adj_set, f)
        print('Saved noun_set and adj_set by pickle\n')

    print('nouns number is %d, adjective number is %d, total nouns and adjectives number is %d\n' % (len(noun_set), len(adj_set), len(noun_set) + len(adj_set)))
    print('First step is finished. Nouns & adjectives were selected.\n')  # for debugging

    # 2. Make noun_pair_set. Find intensifier from the noun
    intensifier_keyword = ['intensifier']
    gutenberg_fdists = dict()
    noun_pair_set = dict()
    total_intensifier_set = dict()

    # Check whether data is produced already or not. Load the data if data is produced.
    if os.path.isfile('gutenberg_fdists.txt') and os.path.isfile('noun_pair_set.txt') and os.path.isfile('total_intensifier_set_nouns.txt'):
        with open('gutenberg_fdists.txt', 'rb') as f:
            gutenberg_fdists = pickle.load(f)
        with open('noun_pair_set.txt', 'rb') as f:
            noun_pair_set = pickle.load(f)
        with open('total_intensifier_set_nouns.txt', 'rb') as f:
            total_intensifier_set = pickle.load(f)
        print("gutenberg_fdists and noun_pair_set, total_intensifier_set data exists. Loaded data\n")

    # If data is not produced yet, start the process. Make noun_pair_set
    else:
        # Create fdist for all of gutenberg corpus
        for each_fileid in gutenberg.fileids():
            each_tokenized_corpus_text = nltk.Text(gutenberg.words(each_fileid))
            gutenberg_fdists[each_fileid] = FreqDist(each_tokenized_corpus_text)

        count = 0
        for each_noun in noun_set:
            # Search in gutenberg corpus
            for each_fileid in gutenberg.fileids():
                each_tokenized_corpus = gutenberg.words(each_fileid)
                for each_index, each_token in enumerate(each_tokenized_corpus):
                    # Find the corresponding noun in corpus
                    if each_token.lower() == each_noun:
                        # Check previous word of given noun
                        if each_index != 0 and each_tokenized_corpus[each_index - 1].isalpha():
                            intensifier = each_tokenized_corpus[each_index - 1].lower()
                            # Find wornet's lemmas and check if there are intensifier_keyword in definition of lemma
                            for synset in wn.synsets(intensifier):
                                found_intensifier = False

                                for each_keyword in intensifier_keyword:
                                    if each_keyword in synset.definition():
                                        # Found intensifier. Keyword is included in the synset definition

                                        # If there is already body word in noun_pair_set, but not found intensifier yet, then add it to existing body word element
                                        if each_noun in noun_pair_set:
                                            if intensifier not in noun_pair_set[each_noun][1]:
                                                # Calculate the frequency of "intensifier body_word"
                                                bigram_frequency = find_frequency(intensifier, each_noun)

                                                # Add it to the noun_pair_set
                                                noun_pair_set[each_noun][1][intensifier] = [bigram_frequency, 0]

                                                # Add it to the total_intensifier_set to calculate frequency for later process
                                                if intensifier in total_intensifier_set:
                                                    if each_noun not in total_intensifier_set[intensifier]:
                                                        total_intensifier_set[intensifier][each_noun] = bigram_frequency

                                                else:
                                                    total_intensifier_set[intensifier] = {each_noun: bigram_frequency}

                                        # If there is no body word in noun_pair_set, then add body word, intensifiier pair into noun_pair_set
                                        else:
                                            # Calculate the frequency of "intensifier body_word"
                                            bigram_frequency = find_frequency(intensifier, each_noun)

                                            # Calculate the frequency of body_word alone
                                            bodyword_frequency = 0
                                            for gutenberg_each_fileid in gutenberg.fileids():
                                                bodyword_frequency += gutenberg_fdists[gutenberg_each_fileid].freq(each_noun)

                                            # Add it to the noun_pair_set
                                            noun_pair_set[each_noun] = [bodyword_frequency, {intensifier: [bigram_frequency, 0]}]

                                            # Add it to the total_intensifier_set to calculate frequency for later process
                                            if intensifier in total_intensifier_set:
                                                if each_noun not in total_intensifier_set[intensifier]:
                                                    total_intensifier_set[intensifier][each_noun] = bigram_frequency

                                            else:
                                                total_intensifier_set[intensifier] = {each_noun: bigram_frequency}

                                        # look for intensifier's lemma's example
                                        for each_example in synset.examples():
                                            tokenized_example = word_tokenize(each_example)
                                            if intensifier in tokenized_example:
                                                next_word_idx = tokenized_example.index(intensifier) + 1
                                                if next_word_idx < len(tokenized_example):
                                                    next_word = tokenized_example[next_word_idx].lower()
                                                    if next_word.isalpha():
                                                        # If there is already body word in noun_pair_set, but not found intensifier yet, then add it to existing body word element
                                                        if next_word in noun_pair_set:
                                                            if intensifier not in noun_pair_set[next_word][1]:
                                                                # Calculate the frequency of "intensifier body_word"
                                                                bigram_frequency = find_frequency(intensifier, next_word)

                                                                # Add it to the noun_pair_set. Add default score 15 since it is found for synset's example
                                                                noun_pair_set[next_word][1][intensifier] = [bigram_frequency, 15]

                                                                # Add it to the total_intensifier_set to calculate frequency for later process
                                                                if intensifier in total_intensifier_set:
                                                                    if next_word not in total_intensifier_set[intensifier]:
                                                                        total_intensifier_set[intensifier][next_word] = bigram_frequency

                                                                else:
                                                                    total_intensifier_set[intensifier] = {next_word: bigram_frequency}

                                                        # If there is no body word in noun_pair_set, then add body word, intensifiier pair into noun_pair_set
                                                        else:
                                                            # Calculate the frequency of "intensifier body_word"
                                                            bigram_frequency = find_frequency(intensifier, next_word)

                                                            # Calculate the frequency of body_word alone
                                                            bodyword_frequency = 0
                                                            for gutenberg_each_fileid in gutenberg.fileids():
                                                                bodyword_frequency += gutenberg_fdists[gutenberg_each_fileid].freq(next_word)

                                                            # Add it to the noun_pair_set. Add default score 15 since it is found for synset's example
                                                            noun_pair_set[next_word] = [bodyword_frequency, {intensifier: [bigram_frequency, 15]}]

                                                            # Add it to the total_intensifier_set to calculate frequency for later process
                                                            if intensifier in total_intensifier_set:
                                                                if next_word not in total_intensifier_set[intensifier]:
                                                                    total_intensifier_set[intensifier][next_word] = bigram_frequency

                                                            else:
                                                                total_intensifier_set[intensifier] = {next_word: bigram_frequency}

                                        found_intensifier = True
                                        break

                                if found_intensifier:
                                    break
            count+=1
            print(str(count) + "\n") # for debugging

        # Save the produced data into file
        with open('gutenberg_fdists.txt', 'wb') as f:
            pickle.dump(gutenberg_fdists, f)
        with open('noun_pair_set.txt', 'wb') as f:
            pickle.dump(noun_pair_set, f)
        with open('total_intensifier_set_nouns.txt', 'wb') as f:
            pickle.dump(total_intensifier_set, f)
        print('Saved gutenberg_fdists, total_intensifier_set_nouns and noun_pair_set by pickle\n')

    print('Second step is finished. noun_pair_set (ditionary) is created and size is %d \n' %(len(noun_pair_set)))  # for debugging

    # 3. Make adj_pair_set. Find intensifier from the adjective
    adj_pair_set = dict()

    # Check whether data is produced already or not. Load the data if data is produced.
    if os.path.isfile('adj_pair_set.txt') and os.path.isfile('total_intensifier_set_nouns_adjs.txt'):
        with open('adj_pair_set.txt', 'rb') as f:
            adj_pair_set = pickle.load(f)
        with open('total_intensifier_set_nouns_adjs.txt', 'rb') as f:
            total_intensifier_set = pickle.load(f)
        print("adj_pair_set and total_intensifier_set_nouns_adjs data exists. Loaded data\n")

    # If data is not produced yet, start the process. Make adj_pair_set
    else:
        count = 0
        for each_adj in adj_set:
            # Search in gutenberg corpus
            for each_fileid in gutenberg.fileids():
                each_tokenized_corpus = gutenberg.words(each_fileid)
                for each_index,each_token in enumerate(each_tokenized_corpus):
                    # Find the corresponding adjective in corpus
                    if each_token.lower() == each_adj:
                        # Check provious word of given adjective
                        if each_index != 0 and each_tokenized_corpus[each_index-1].isalpha():
                            intensifier = each_tokenized_corpus[each_index - 1].lower()
                            # Find wornet's lemmas and check if there are intensifier_keyword in definition of lemma
                            for synset in wn.synsets(intensifier):
                                found_intensifier = False

                                for each_keyword in intensifier_keyword:
                                    if each_keyword in synset.definition():
                                        # Found intensifier. Keyword is included in the synset definition

                                        # If there is already body word in adj_pair_set, but not found intensifier yet, then add it to existing body word element
                                        if each_adj in adj_pair_set:
                                            if intensifier not in adj_pair_set[each_adj][1]:
                                                # Calculate the frequency of "intensifier body_word"
                                                bigram_frequency = find_frequency(intensifier, each_adj)

                                                # Add it to the adj_pair_set
                                                adj_pair_set[each_adj][1][intensifier] = [bigram_frequency, 0]

                                                # Add it to the total_intensifier_set to calculate frequency for later process
                                                if intensifier in total_intensifier_set:
                                                    if each_adj not in total_intensifier_set[intensifier]:
                                                        total_intensifier_set[intensifier][each_adj] = bigram_frequency

                                                else:
                                                    total_intensifier_set[intensifier] = {each_adj: bigram_frequency}

                                        # If there is no body word in adj_pair_set, then add body word, intensifiier pair into adj_pair_set
                                        else:
                                            # Calculate the frequency of "intensifier body_word"
                                            bigram_frequency = find_frequency(intensifier, each_adj)

                                            # Calculate the frequency of body_word alone
                                            bodyword_frequency = 0
                                            for gutenberg_each_fileid in gutenberg.fileids():
                                                bodyword_frequency += gutenberg_fdists[gutenberg_each_fileid].freq(each_adj)

                                            # Add it to the adj_pair_set
                                            adj_pair_set[each_adj] = [bodyword_frequency, {intensifier: [bigram_frequency, 0]}]

                                            # Add it to the total_intensifier_set to calculate frequency for later process
                                            if intensifier in total_intensifier_set:
                                                if each_adj not in total_intensifier_set[intensifier]:
                                                    total_intensifier_set[intensifier][each_adj] = bigram_frequency

                                            else:
                                                total_intensifier_set[intensifier] = {each_adj: bigram_frequency}

                                        # look for intensifier's lemma's example
                                        for each_example in synset.examples():
                                            tokenized_example = word_tokenize(each_example)
                                            if intensifier in tokenized_example:
                                                next_word_idx = tokenized_example.index(intensifier) + 1
                                                if next_word_idx < len(tokenized_example):
                                                    next_word = tokenized_example[next_word_idx].lower()
                                                    if next_word.isalpha():
                                                        # If there is already body word in adj_pair_set, but not found intensifier yet, then add it to existing body word element
                                                        if next_word in adj_pair_set:
                                                            if intensifier not in adj_pair_set[next_word][1]:
                                                                # Calculate the frequency of "intensifier body_word"
                                                                bigram_frequency = find_frequency(intensifier, next_word)

                                                                # Add it to the adj_pair_set. Add default score 15 since it is found for synset's example
                                                                adj_pair_set[next_word][1][intensifier] = [bigram_frequency, 15]

                                                                # Add it to the total_intensifier_set to calculate frequency for later process
                                                                if intensifier in total_intensifier_set:
                                                                    if next_word not in total_intensifier_set[intensifier]:
                                                                        total_intensifier_set[intensifier][next_word] = bigram_frequency

                                                                else:
                                                                    total_intensifier_set[intensifier] = {next_word: bigram_frequency}

                                                        # If there is no body word in adj_pair_set, then add body word, intensifiier pair into adj_pair_set
                                                        else:
                                                            # Calculate the frequency of "intensifier body_word"
                                                            bigram_frequency = find_frequency(intensifier, next_word)

                                                            # Calculate the frequency of body_word alone
                                                            bodyword_frequency = 0
                                                            for gutenberg_each_fileid in gutenberg.fileids():
                                                                bodyword_frequency += gutenberg_fdists[gutenberg_each_fileid].freq(next_word)

                                                            # Add it to the adj_pair_set. Add default score 15 since it is found for synset's example
                                                            adj_pair_set[next_word] = [bodyword_frequency, {intensifier: [bigram_frequency, 15]}]

                                                            # Add it to the total_intensifier_set to calculate frequency for later process
                                                            if intensifier in total_intensifier_set:
                                                                if next_word not in total_intensifier_set[intensifier]:
                                                                    total_intensifier_set[intensifier][next_word] = bigram_frequency

                                                            else:
                                                                total_intensifier_set[intensifier] = {next_word: bigram_frequency}

                                        found_intensifier = True
                                        break

                                if found_intensifier:
                                    break
            count+=1
            print(str(count) + "\n") # for debugging

        # Save the produced data into file
        with open('adj_pair_set.txt', 'wb') as f:
            pickle.dump(adj_pair_set, f)
        with open('total_intensifier_set_nouns_adjs.txt', 'wb') as f:
            pickle.dump(total_intensifier_set, f)
        print('Saved adj_pair_set and total_intensifier_set_nouns_adjs by pickle\n')

    print('Third step is finished. adj_pair_set (ditionary) is created and size is %d \n' % (len(adj_pair_set)))  # for debugging

    # 4. Calculate Score and combine nouns_pair_set, ajd_pair_set into total_pair_set
    total_pair_set = []

    # Check whether data is produced already or not. Load the data if data is produced.
    if os.path.isfile('adj_scored_pair_set.txt') and os.path.isfile('noun_scored_pair_set.txt') and os.path.isfile('total_pair_set.txt'):
        with open('adj_scored_pair_set.txt', 'rb') as f:
            adj_pair_set = pickle.load(f)
        with open('noun_scored_pair_set.txt', 'rb') as f:
            noun_pair_set = pickle.load(f)
        with open('total_pair_set.txt', 'rb') as f:
            total_pair_set = pickle.load(f)
        print("adj_scored_pair_set, noun_socred_pair_set and total_pair_set data exists. Loaded data\n")

    # If data is not produced yet, start the process. Calculate the score
    else:
        # Calculate score for nouns
        count = 0
        for each_body_word, body_word_data in noun_pair_set.items():
            body_word_frequency = body_word_data[0]
            for each_intensifier, intensifier_data in body_word_data[1].items():
                bigram_frequency = intensifier_data[0]

                # Calculate the average frequency of intensifiers from other body words
                frequency_sum = 0
                freqeuncy_count = 0
                for each_body_word_2, each_body_word_frequency_2 in total_intensifier_set[each_intensifier].items():
                    if each_body_word_2 != each_body_word:
                        freqeuncy_count += 1
                        frequency_sum += each_body_word_frequency_2
                frequency_avg = 0
                if freqeuncy_count != 0:
                    frequency_avg = frequency_sum/freqeuncy_count

                # Calculate the score and save it
                score = calculate_score(body_word_frequency, bigram_frequency, frequency_avg)
                intensifier_data[1] += score

                # Add data into total_pair_set
                total_pair_set.append((each_intensifier, each_body_word, intensifier_data[1]))

            count+=1
            print(str(count) + "\n") # for debugging

        print('Noun score calcuating finish\n') # for dubugging

        # Calculate score for adjectives
        count=0
        for each_body_word, body_word_data in adj_pair_set.items():
            body_word_frequency = body_word_data[0]
            for each_intensifier, intensifier_data in body_word_data[1].items():
                bigram_frequency = intensifier_data[0]

                # Calculate the average frequency of intensifiers from other body words
                frequency_sum = 0
                freqeuncy_count = 0
                for each_body_word_2, each_body_word_frequency_2 in total_intensifier_set[each_intensifier].items():
                    if each_body_word_2 != each_body_word:
                        freqeuncy_count += 1
                        frequency_sum += each_body_word_frequency_2
                frequency_avg = 0
                if freqeuncy_count != 0:
                    frequency_avg = frequency_sum/freqeuncy_count

                # Calculate the score and save it
                score = calculate_score(body_word_frequency, bigram_frequency, frequency_avg)
                intensifier_data[1] += score

                # Add data into total_pair_set
                total_pair_set.append((each_intensifier, each_body_word, intensifier_data[1]))

            count+=1
            print(str(count) + "\n") # for debugging

        print('Adjective score calcuating finish\n') # for debugging

        # Sort the total_pair_set by score
        total_pair_set = set(total_pair_set)
        total_pair_set = sorted(total_pair_set, key=lambda each_pair: each_pair[2], reverse=True)
        print('Total pair set sorting finish\n') # for debugging

        # Save the produced data into file
        with open('adj_scored_pair_set.txt', 'wb') as f:
            pickle.dump(adj_pair_set, f)
        with open('noun_scored_pair_set.txt', 'wb') as f:
            pickle.dump(noun_pair_set, f)
        with open('total_pair_set.txt', 'wb') as f:
            pickle.dump(total_pair_set, f)
        print('Saved adj, nouns scored sets and total_pair_set by pickle\n')

    print('Fourth step is finished. adj_pair_set and noun_pair_set is scored, and total_pair_set is created and sorted \n')  # for debugging

    # 5. Print out result in csv file
    hundread_selected_result = total_pair_set[:100]
    f = open('result.csv', 'w') # make "result.csv" file
    for each_pair in hundread_selected_result:
        f.write(each_pair[0] + ',' + each_pair[1] + '\n')
    print('Fifth step is finished. Selected 100 most unique intensifier-body word sets and wrote it on csv file \n')  # for debugging



def calculate_score(body_word_frequency, target_bigram_frequency, others_bigram_frequency_avg):
    # 1. total 30 points part. Check how meaning of bigram is strong by checking the frequency
    score_1 = 0
    if body_word_frequency != 0:
        x = target_bigram_frequency/body_word_frequency
        score_1 = 30*(1-x)

    # 2. total 70 points part. Check how "intensifier body_word" frequency is restrictive than "intensifier and other body_words"
    score_2 = 0
    if target_bigram_frequency != 0:
        y = others_bigram_frequency_avg/target_bigram_frequency
        score_2 = 70*(1-y)

    return score_1+score_2



def find_frequency(intensifier, body_word):
    # Search in gutenberg corpus
    count = 0

    for each_fileid in gutenberg.fileids():
        each_text = gutenberg.raw(each_fileid)
        count += each_text.count(intensifier + " " + body_word)
        count += each_text.count(intensifier + "-" + body_word)

    return count


get_restrictive_intensifier()