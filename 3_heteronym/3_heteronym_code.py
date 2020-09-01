import nltk
import random
import pickle
import os
import requests
import json
import numpy
from nltk.corpus import *
# from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import FreqDist

def make_heteronym_list():
    wnl = nltk.WordNetLemmatizer()
    prondict = nltk.corpus.cmudict.dict()

    # 1. Make candidates of heteronyms which is calculated by using wordnet and cmudict
    # Check whether data is produced already or not. Load the data if data is produced.
    if os.path.isfile('heteronym_candidates.txt'):
        with open('heteronym_candidates.txt', 'rb') as f:
            heteronym_candidates = pickle.load(f)
        print("heteronym candidates exists. Loaded data\n")

    # If data is not produced yet, start the process. Make heteronym_candidates
    else:
        # Get word set of guteberg corpus
        raw_word_set = set([each_word.lower() for each_fileid in gutenberg.fileids() for each_word in
                            set(gutenberg.words(each_fileid))])
        raw_word_set = pos_tag(raw_word_set)

        # Match nltk's part-of-speech symbols with wordNet part-of-speech symbols and lemmatize the words
        lemmatizated_word_set  = set()
        for each_word, each_pos in raw_word_set:
            if each_pos == "NN" or each_pos == "NNS": # noun
                lemmatizated_word_set.add(wnl.lemmatize(each_word, 'n').lower())

            elif each_pos == "VB" or each_pos == "VBD" or each_pos == "VBG" or each_pos == "VBN" or each_pos == "VBP" or each_pos == "VBZ": # verb
                lemmatizated_word_set.add(wnl.lemmatize(each_word, 'v').lower())

            elif each_pos == "JJ" or each_pos == "JJR" or each_pos == "JJS": # adjective
                lemmatizated_word_set.add(wnl.lemmatize(each_word, 'a').lower())

            elif each_pos == "RB" or each_pos == "RBR" or each_pos == "RBS": # adverb
                lemmatizated_word_set.add(wnl.lemmatize(each_word, 'r').lower())


        # Check each word's searching result in prondict is larger than 1 and synnonyms number is also larger than 1. That would be heteronym candidate
        heteronym_candidates = []
        for each_word in lemmatizated_word_set:
            if each_word in prondict:
                # If prondict's result's length is larger or equal than 2, it means there can be many pronunciations
                if len(prondict[each_word]) >= 2:
                    each_poses = dict()

                    # Search synset's name in the wordNet and exclude the last num part (ex - .02)
                    # only consider first elements of synsets with same name (ex - car.n.01, car.n.02 only pick car.n.01)
                    # since it would be the most frequent meaning
                    for each_synset in wn.synsets(each_word):
                        if each_synset.pos() not in each_poses:
                            each_poses[each_synset.pos()] = {each_synset.name()[:-3]: each_synset.definition()}
                        else:
                            if each_synset.name()[:-3] not in each_poses[each_synset.pos()]:
                                each_poses[each_synset.pos()][each_synset.name()[:-3]] = each_synset.definition()

                    # Check whether picked synsets numbers are more or eqaul than 2
                    if len(each_poses)>0:
                        total_num = 0
                        for each_pos in each_poses:
                            total_num += len(each_pos)
                        if total_num >= 2:
                            heteronym_candidates.append(each_word)

        # Save the produced data into file
        with open('heteronym_candidates.txt', 'wb') as f:
            pickle.dump(heteronym_candidates, f)
        print('Saved heteronym candidates by pickle\n') # for debugging

    print('First step is finished. Heteronym candidates is produced and length is %d \n' %(len(heteronym_candidates))) # for debugging
    # print(heteronym_candidates) # for debugging

    # 2. Make heteronym dictiony. Select some of candidates from heteronym_candidates by oxvford dictionary
    # Check whether data is produced already or not. Load the data if data is produced.
    if os.path.isfile('heteronym_dict.txt'):
        with open('heteronym_dict.txt', 'rb') as f:
            heteronym_dict = pickle.load(f)
        print("heteronym dictionary exists. Loaded data\n")

    else:
        # Oxford dictionary rest API info
        app_id = "18c7e5a9"
        app_key = "c12e9270cfb629b69ffcc183d2df71da"
        endpoint = "entries"
        language_code = "en-us"

        # Make heteronyms dictionary by selecting heteronym in its candidates list using oxford dictionary
        heteronym_dict = dict()
        for each_count, each_candidate in enumerate(heteronym_candidates):
            print(each_count) # for debugging
            word_id = each_candidate
            url = "https://od-api.oxforddictionaries.com/api/v2/" + endpoint + "/" + language_code + "/" + word_id.lower() + '?' \
                  + 'fields=definitions%2Cpronunciations&strictMatch=false'
            response = requests.get(url, headers={"app_id": app_id, "app_key": app_key})
            result_json = response.json()

            # Work only http response replied succesfully
            if response.status_code != 200:
                continue

            # By parsing json file, analysis word's definition with it's pronunciation
            each_dict = dict()
            each_pron_set = set()
            if "results" in result_json:
                for each_result in result_json["results"]:
                    if "lexicalEntries" not in each_result:
                        continue
                    for each_lexical_entry in each_result["lexicalEntries"]:  # each_lexical_entry is dictionary
                        if each_lexical_entry["lexicalCategory"]["id"] not in each_dict:
                            count = 0
                            definition = ""
                            pron_ipa = ""
                            if "pronunciations" in each_lexical_entry:
                                for each_pron in each_lexical_entry["pronunciations"]:
                                    if each_pron.get("phoneticNotation") == "IPA":
                                        pron_ipa = each_pron.get("phoneticSpelling")
                                        break
                            if "entries" not in each_lexical_entry:
                                continue
                            for each_entry in each_lexical_entry["entries"]:
                                if "pronunciations" in each_entry:
                                    for each_pron in each_entry["pronunciations"]:
                                        if each_pron.get("phoneticNotation") == "IPA":
                                            pron_ipa = each_pron.get("phoneticSpelling")
                                            break
                                if "senses" in each_entry:
                                    for each_sense in each_entry["senses"]:
                                        count += 1
                                        definition = each_sense.get("definitions")[0]
                                        if "pronunciations" in each_sense:
                                            for each_pron in each_sense["pronunciations"]:
                                                if each_pron.get("phoneticNotation") == "IPA":
                                                    pron_ipa = each_pron.get("phoneticSpelling")
                                                    break
                                        if pron_ipa != None:
                                            # Initial case for adding ipa and definitiona data into dictionary
                                            if count == 1:
                                                each_dict[each_lexical_entry["lexicalCategory"]["id"]] = [(pron_ipa, definition)]
                                                each_pron_set.add(pron_ipa)
                                            # Add ipa and definitiona data into dictioanry
                                            else:
                                                each_dict[each_lexical_entry["lexicalCategory"]["id"]].append(
                                                    (pron_ipa, definition))
                                                each_pron_set.add(pron_ipa)
                        else:
                            definition = ""
                            pron_ipa = ""
                            if "pronunciations" in each_lexical_entry:
                                for each_pron in each_lexical_entry["pronunciations"]:
                                    if each_pron.get("phoneticNotation") == "IPA":
                                        pron_ipa = each_pron.get("phoneticSpelling")
                                        break
                            if "entries" not in each_lexical_entry:
                                continue
                            for each_entry in each_lexical_entry["entries"]:
                                if "pronunciations" in each_entry:
                                    for each_pron in each_entry["pronunciations"]:
                                        if each_pron.get("phoneticNotation") == "IPA":
                                            pron_ipa = each_pron.get("phoneticSpelling")
                                            break
                                if "senses" in each_entry:
                                    for each_sense in each_entry["senses"]:
                                        definition = each_sense.get("definitions")[0]
                                        if "pronunciations" in each_sense:
                                            for each_pron in each_sense["pronunciations"]:
                                                if each_pron.get("phoneticNotation") == "IPA":
                                                    pron_ipa = each_pron.get("phoneticSpelling")
                                                    break
                                        # Add ipa and definitiona data into dictioanry
                                        if pron_ipa != None:
                                            each_dict[each_lexical_entry["lexicalCategory"]["id"]].append((pron_ipa, definition))
                                            each_pron_set.add(pron_ipa)

            # print(each_candidate) # for debugging
            # When (pronuncation annotation, definition) data's number is larger than 1, it means it's heteronym
            # since it has different meaning on different pronunciation in same spelling
            # otherwise candidate is not heteronym. So discard those results
            if len(each_pron_set)>1:
                # print((each_dict, each_pron_set)) # for debugging
                heteronym_dict[each_candidate] = (each_dict, each_pron_set)

        # Exclude be for better result
        if 'be' in heteronym_dict:
            del heteronym_dict['be']

        # Save the produced data into file
        with open('heteronym_dict.txt', 'wb') as f:
            pickle.dump(heteronym_dict, f)
        print('Saved heteronym dictionary by pickle\n') # for debugging

    print('Second step is finished. Heteronym dictionary is produced and length is %d \n' %(len(heteronym_dict))) # for debugging
    # print(heteronym_dict) # for debugging

    # 3. Calculate score for each corpora. It will helps to match word's pronunciation for latter process
    corpus_writing_score = dict()

    # Check whether data is produced already or not. Load the data if data is produced.
    if os.path.isfile('corpus_writing_score.txt'):
        with open('corpus_writing_score.txt', 'rb') as f:
            corpus_writing_score = pickle.load(f)
        print("corpus writing score dictionary exists. Loaded data\n")

    else:
        # There are five categories
        sentence_length_var_all = dict() # 1. corpus var of sentence length
        sentence_word_count_all = dict() # 2. corpus abs(diff of average sentece's words number with 17)
        word_set_len_all = dict() # 3. corpus avg length of word set
        word_len_avg_all = dict() # 4. corpus avg length of words
        verb_count_all = dict() # 5. corpus avg numbers of verbs per sentence

        # Start to calculate corpus score
        print("start calcuating corpus writing score\n") # for debugging
        for each_index, each_fileid in enumerate(gutenberg.fileids()):
            print(each_index) # for debugging
            each_corpus_sents_length = []
            each_corpus_sents_word_count = []
            each_corpus_words_length = []
            each_sents_verb_count = []

            print('process 1') # for debugging
            for each_sent in gutenberg.sents(each_fileid):
                each_corpus_sents_word_count.append(len(each_sent)) # count the number of words of each sentence
                sent_length = 0
                each_verb_count = 0

                for each_word, each_pos in pos_tag(each_sent):
                    sent_length+=len(each_word) # count sentence total length
                    if each_pos == "VB" or each_pos == "VBD" or each_pos == "VBG" or each_pos == "VBN" or each_pos == "VBP" or each_pos == "VBZ":
                        each_verb_count += 1 # count the verb's number in sentence

                each_corpus_sents_length.append(sent_length)
                each_sents_verb_count.append(each_verb_count)

            print('process 2') # for debugging
            for each_word in gutenberg.words(each_fileid):
                each_corpus_words_length.append(len(each_word)) # count word's length

            print('process 3') # for debugging
            sentence_length_var_all[each_fileid] = numpy.var(each_corpus_sents_length)
            sentence_word_count_all[each_fileid] = abs(17 - numpy.mean(each_corpus_sents_word_count))
            word_set_len_all[each_fileid] = len(set(gutenberg.words(each_fileid)))
            word_len_avg_all[each_fileid] = numpy.mean(each_corpus_words_length)
            verb_count_all[each_fileid] = numpy.mean(each_sents_verb_count)

        sentence_length_var_all_sorted = sorted(sentence_length_var_all.values()) # 1.
        sentence_word_count_all_sorted = sorted(sentence_word_count_all.values()) # 2.
        word_set_len_all_sorted = sorted(word_set_len_all.values()) # 3.
        word_len_avg_all_sorted = sorted(word_len_avg_all.values()) # 4.
        verb_count_all_sorted = sorted(verb_count_all.values()) # 5.

        corpus_writing_score_pre = dict()
        for each_index, each_fileid in enumerate(gutenberg.fileids()):
            each_corpus_score = 0

            # Sorting
            # 1.
            each_corpus_score += (sentence_length_var_all[each_fileid]/sentence_length_var_all_sorted[-1])*20
            # 2.
            each_corpus_score += (1 - sentence_word_count_all[each_fileid]/sentence_word_count_all_sorted[-1])*20
            # 3.
            each_corpus_score += (word_set_len_all[each_fileid]/word_set_len_all_sorted[-1])*20
            # 4.
            each_corpus_score += (word_len_avg_all[each_fileid]/word_len_avg_all_sorted[-1])*20
            # 5.
            each_corpus_score += (verb_count_all[each_fileid]/verb_count_all_sorted[-1])*20

            corpus_writing_score_pre[each_fileid] = each_corpus_score

        corpus_writing_score_pre = sorted(corpus_writing_score_pre.items(), key=lambda item: item[1])

        # Match the cursor value by orders
        for each_index, each_item in enumerate(corpus_writing_score_pre):
            corpus_writing_score[each_item[0]] = each_index+1

        # Save the produced data into file
        with open('corpus_writing_score.txt', 'wb') as f:
            pickle.dump(corpus_writing_score, f)
        print('Saved corpus writing score dictionary by pickle\n') # for debugging

    print('Third step is finished. Corpus writing scores are produced\n') # for debugging
    # print(corpus_writing_score) # for debugging

    # 4. Find sentences with heteronym in corpus
    heteronym_sents =[]
    # Check whether data is produced already or not. Load the data if data is produced.
    if os.path.isfile('heteronym_sents.txt'):
        with open('heteronym_sents.txt', 'rb') as f:
            heteronym_sents = pickle.load(f)
        print("heteronym sentence exists. Loaded data\n")

    else:
        for each_index, each_fileid in enumerate(gutenberg.fileids()):
            print("{} guteberberg corpus processing".format(each_index+1)) # for debugging
            each_corpus_score_order = corpus_writing_score[each_fileid]

            # Sentence segmentation
            for each_sent in gutenberg.sents(each_fileid):
                each_pos_tagged_sent = pos_tag(each_sent) # pos tagging
                each_pos_lemmatized_tagged_sent = []

                # Lemmatizing and tagging
                for each_item in each_pos_tagged_sent:
                    if each_item[1] == "NN" or each_item[1] == "NNS": # noun
                        each_pos_lemmatized_tagged_sent.append([each_item[0], each_item[1], wnl.lemmatize(each_item[0], 'n').lower()])

                    elif each_item[1] == "VB" or each_item[1] == "VBD" or each_item[1] == "VBG" or each_item[1] == "VBN" or each_item[1] == "VBP" or each_item[1] == "VBZ": # verb
                        each_pos_lemmatized_tagged_sent.append([each_item[0], each_item[1], wnl.lemmatize(each_item[0], 'v').lower()])

                    elif each_item[1] == "JJ" or each_item[1] == "JJR" or each_item[1] == "JJS": # adjective
                        each_pos_lemmatized_tagged_sent.append([each_item[0], each_item[1], wnl.lemmatize(each_item[0], 'a').lower()])

                    elif each_item[1] == "RB" or each_item[1] == "RBR" or each_item[1] == "RBS": # adverb
                        each_pos_lemmatized_tagged_sent.append([each_item[0], each_item[1], wnl.lemmatize(each_item[0], 'r').lower()])

                appeared_heteronyms_dict = dict()
                appeared_heteronyms = []
                each_score = 0

                # Calculating score and find heteronyms in the sentence
                for each_item in each_pos_lemmatized_tagged_sent:
                    oxf_formatted_pos = change_pos_format_nltk_to_oxf(each_item[1]) # change spos symbol to oxford dictionary format
                    # each_item[0] is original word, each_item[1] is pos(nltk), each_item[2] is lemmatized word

                    # If lemmatized word of sentence is heteronym, we should consider this sentence for element of heteronym_sents
                    # Check lemmatized word is in heteronym dictioanry
                    if each_item[2] in heteronym_dict:
                        # Check same spelling's lemmatized heteronym was in the same sentence
                        # if not, we can just simply add it into appeared_heteronyms without considering making the pair
                        # we can just simply add score 1, and save those data into appeared_heteronyms_dict also to check same spelling's heteronym for later case
                        if each_item[2] not in appeared_heteronyms_dict:
                            # Get randomized 0 or 1 value to decide use most ferquent definition with proper pos, or latter one
                            dice = random.randrange(0, 2)

                            # If dice vaule is 0, we will use most frequent definition with proper pos
                            if dice == 0:
                                each_pron = ""
                                each_def =  ""

                                # Search heteronym_dict with lemmatized word, and search corresponding pos exists or not
                                # ignore if searched_pos doesn't exist in heteronym_dict. Pos tagging might worked wrong
                                if heteronym_dict[each_item[2]][0].get(oxf_formatted_pos) != None:
                                    # Get pronunciation annontation and definition information from heteronym dictioanry
                                    each_pron = heteronym_dict[each_item[2]][0].get(oxf_formatted_pos)[0][0]
                                    each_def = heteronym_dict[each_item[2]][0].get(oxf_formatted_pos)[0][1]
                                    appeared_heteronyms_dict[each_item[2]] = {oxf_formatted_pos: [(each_pron, each_def)]} # add it to appeared_heteronyms_dict to find pair
                                    appeared_heteronyms.append((each_item[0], each_item[2], oxf_formatted_pos,each_pron,each_def)) # add it to appeared_heteronyms to make csv file
                                    each_score += 1 # heteronym occured. plus one point

                            # If dice value is 1, we will use less frequent deifinition with proper pos
                            else:
                                each_pron = ""
                                each_def =  ""

                                # Search heteronym_dict with lemmatized word, and search corresponding pos exists or not
                                # ignore if searched_pos doesn't exist in heteronym_dict. Pos_tagging might worked wrong
                                if heteronym_dict[each_item[2]][0].get(oxf_formatted_pos) != None:
                                    # Calculate the total (pron, definition) item numbers in corresponding pos category in heteronym dict
                                    dict_pos_total_num = len(heteronym_dict[each_item[2]][0].get(oxf_formatted_pos))

                                    # Calculate the cursor which indicates what less frequent definition should we use
                                    cursor = int(dict_pos_total_num * (each_corpus_score_order/18))

                                    # Considering cursor's maximum case
                                    if cursor > dict_pos_total_num-1:
                                        cursor = dict_pos_total_num - 1

                                    # Get pronunciation annontation and definition information from heteronym dictioanry
                                    each_pron = heteronym_dict[each_item[2]][0].get(oxf_formatted_pos)[cursor][0]
                                    each_def = heteronym_dict[each_item[2]][0].get(oxf_formatted_pos)[cursor][1]
                                    appeared_heteronyms_dict[each_item[2]] = {oxf_formatted_pos: [(each_pron, each_def)]}  # add it to appeared_heteronyms_dict to find pair
                                    appeared_heteronyms.append((each_item[0], each_item[2], oxf_formatted_pos, each_pron, each_def)) # add it to appeared_heteronyms to make csv file
                                    each_score += 1

                        # Check same spelling's lemmatized heteronym was in the same sentence
                        # if yes, we have to consider pair is made or not. Pair means heteronym pair(ex- bass(adj), bass(n)) and if pair is made, then check two word's pos is same or not
                        # Add score 1, and additional pair score 49.9 or 50 by pos. And save those data into appeared_heteronyms_dict also to check same spelling's heteronym for later case
                        else:
                            # Get randomized 0 or 1 value to decide use most ferquent definition with proper pos, or latter one
                            dice = random.randrange(0, 2)

                            # If dice vaule is 0, we will use most frequent definition with proper pos
                            if dice == 0:
                                each_pron = ""
                                each_def =  ""

                                # Search heteronym_dict with lemmatized word, and search corresponding pos exists or not
                                # ignore if searched_pos doesn't exist in heteronym_dict. Pos_tagging might worked wrong
                                if heteronym_dict[each_item[2]][0].get(oxf_formatted_pos) != None:
                                    # Get pronunciation annontation and definition information from heteronym dictioanry
                                    each_pron = heteronym_dict[each_item[2]][0].get(oxf_formatted_pos)[0][0]
                                    each_def = heteronym_dict[each_item[2]][0].get(oxf_formatted_pos)[0][1]

                                    # Check it makes pair or not. and pos is differnt or not. If it makes pair, give bonus score 50 or 49.9 by comparing pos
                                    its_not_pair = False
                                    diff_pos = False

                                    # Check if this word's pos is different from all other previous appeared same spelling's heteronyms.
                                    if appeared_heteronyms_dict[each_item[2]].get(oxf_formatted_pos) == None:
                                        diff_pos = True

                                    if not diff_pos:
                                        for already_appeared_pron, already_appeared_def in appeared_heteronyms_dict[each_item[2]][oxf_formatted_pos]:
                                            # Check whether there is same tuple in proper pos category in appeared_heteronyms_dict
                                            # if it exists, it's not making pair
                                            if already_appeared_pron == each_pron or already_appeared_def == each_def:
                                                its_not_pair = True
                                                break

                                    else:
                                        for each_oxf_foramtted_pos in appeared_heteronyms_dict[each_item[2]]:
                                            for already_appeared_pron, already_appeared_def in appeared_heteronyms_dict[each_item[2]][each_oxf_foramtted_pos]:
                                                # Check whether there is same tuple in appeared_heteronyms_dict
                                                # if it exists, it's not making pair
                                                if already_appeared_pron == each_pron or already_appeared_def == each_def:
                                                    its_not_pair = True
                                                    break
                                            if its_not_pair:
                                                break

                                    # If it makes pair, we have to apply bonus pair point 50 or 49.9
                                    if not its_not_pair:
                                        # Check pos is same or not to apply 50 or 49.9 pair point
                                        # when pos is diffenent +49.9 point
                                        if diff_pos:
                                            each_score += 49.9
                                        # when pos is same + 50 point
                                        else:
                                            each_score += 50

                                    if not diff_pos:
                                        appeared_heteronyms_dict[each_item[2]][oxf_formatted_pos].append((each_pron, each_def))  # add it to appeared_heteronyms_dict to find pair
                                    else:
                                        appeared_heteronyms_dict[each_item[2]][oxf_formatted_pos] = [(each_pron, each_def)]  # add it to appeared_heteronyms_dict to find pair

                                    appeared_heteronyms.append((each_item[0], each_item[2], oxf_formatted_pos,each_pron,each_def)) # add it to appeared_heteronyms to make csv file
                                    each_score += 1 # heteronym occured. plus one point

                            # If dice value is 1, we will use less frequent deifinition with proper pos
                            else:
                                each_pron = ""
                                each_def =  ""

                                # Search heteronym_dict with lemmatized word, and search corresponding pos exists or not
                                # ignore if searched_pos doesn't exist in heteronym_dict. Pos_tagging might worked wrong
                                if heteronym_dict[each_item[2]][0].get(oxf_formatted_pos) != None:
                                    # calculate the total (pron, definition) item number in corresponding pos category in heteronym dict
                                    dict_pos_total_num = len(heteronym_dict[each_item[2]][0].get(oxf_formatted_pos))

                                    # calculate the cursor which indicates what less frequent definition should we use
                                    cursor = int(dict_pos_total_num * (each_corpus_score_order/18))

                                    # Considering cursor's maximum case
                                    if cursor > dict_pos_total_num-1:
                                        cursor = dict_pos_total_num - 1

                                    # Get pronunciation annontation and definition information from heteronym dictioanry
                                    each_pron = heteronym_dict[each_item[2]][0].get(oxf_formatted_pos)[cursor][0]
                                    each_def = heteronym_dict[each_item[2]][0].get(oxf_formatted_pos)[cursor][1]

                                    # Check it makes pair or not. and pos is differnt or not. If it makes pair, give bonus score 50 or 49.9 by comparing pos
                                    its_not_pair = False
                                    diff_pos = False

                                    # Check if this word's pos is different from all other previous appeared same spelling's heteronyms.
                                    if appeared_heteronyms_dict[each_item[2]].get(oxf_formatted_pos) == None:
                                        diff_pos = True

                                    if not diff_pos:
                                        for already_appeared_pron, already_appeared_def in appeared_heteronyms_dict[each_item[2]][oxf_formatted_pos]:
                                            # Check whether there is same tuple in proper pos category in appeared_heteronyms_dict
                                            # if it exists, it's not making pair
                                            if already_appeared_pron == each_pron or already_appeared_def == each_def:
                                                its_not_pair = True
                                                break

                                    else:
                                        for each_oxf_foramtted_pos in appeared_heteronyms_dict[each_item[2]]:
                                            for already_appeared_pron, already_appeared_def in appeared_heteronyms_dict[each_item[2]][each_oxf_foramtted_pos]:
                                                # Check whether there is same tuple in proper pos category in appeared_heteronyms_dict
                                                # if it exists, it's not making pair
                                                if already_appeared_pron == each_pron or already_appeared_def == each_def:
                                                    its_not_pair = True
                                                    break
                                            if its_not_pair:
                                                break

                                    # If it makes pair, we have to apply bonus pair point 50 or 49.9
                                    if not its_not_pair:
                                        # Check pos is same or not to apply 50 or 49.9 pair point
                                        # when pos is diffenent +49.9 point
                                        if diff_pos:
                                            each_score += 49.9
                                        # when pos is same + 50 point
                                        else:
                                            each_score += 50

                                    if not diff_pos:
                                        appeared_heteronyms_dict[each_item[2]][oxf_formatted_pos].append((each_pron, each_def))  # add it to appeared_heteronyms_dict to find pair
                                    else:
                                        appeared_heteronyms_dict[each_item[2]][oxf_formatted_pos] = [(each_pron, each_def)]  # add it to appeared_heteronyms_dict to find pair

                                    appeared_heteronyms.append((each_item[0], each_item[2], oxf_formatted_pos, each_pron, each_def)) # add it to appeared_heteronyms to make csv file
                                    each_score += 1 # heteronym occured. plus one point

                # Only considet sentences which heteronyms appeared at least once
                if len(appeared_heteronyms)>0:
                    each_heteronym_sents_elem = [each_sent, each_score, "gutenberg corpus", appeared_heteronyms]
                    heteronym_sents.append(each_heteronym_sents_elem)

        print("Sentence processing finish. heteronym_sents sorting by score start") # for debugging
        heteronym_sents = sorted(heteronym_sents, key=lambda each_elem: each_elem[1] ,reverse=True) # sorting by score
        print("heteronym_sents sorting finished") # for debugging

        # Save the produced data into file
        with open('heteronym_sents.txt', 'wb') as f:
            pickle.dump(heteronym_sents, f)
        print('Saved heteronym sentence lists by pickle\n')

    print('Fourth step is finished. heteronym sentences are produced and length is {}\n'.format(len(heteronym_sents))) # for debugging

    # 5. Print out result in csv file
    thirty_selected_result = heteronym_sents[:30]
    f = open('result.csv', 'w', -1, 'utf-8') # make "result.csv" file
    f.write('sentence,citation,heteronyms(lemmatized ver)(pronunciation)' + '\n') # for categorization
    for each_item in thirty_selected_result:
        for each_word in each_item[0]: # print whole sentece

            # !!!!!! IMPORTANT !!!!!!
            # !!!!!! IMPORTANT !!!!!!
            # !!!!!! IMPORTANT !!!!!!
            # Since comma(,) and double quote(") breaks the format and makes confusion in csv file format,
            # I changed them into (comma) (double quotes) with these phrases instead. Consider it please
            # !!!!!! IMPORTANT !!!!!!
            # !!!!!! IMPORTANT !!!!!!
            # !!!!!! IMPORTANT !!!!!!

            if each_word == ',':
                f.write('(comma) ') # considering ',' comma in sentence
            elif each_word == "\"":
                f.write('(double quotes) ')  # considering '"' double quotes in sentence
            else:
                f.write(each_word + ' ')
        f.write(',')
        f.write(each_item[2] + ',') # print citation
        for each_heteronym_data in each_item[3]: # print heteronyms and their information
            f.write("{}({})({})".format(each_heteronym_data[0], each_heteronym_data[1], each_heteronym_data[3]))
            f.write(',')
        f.write('\n')
    print('Fifth step is finished. Selected 30 most high rank sentences with heteronyms \n')  # for debugging

def change_pos_format_nltk_to_oxf(target_input_pos):
    if target_input_pos == 'NN' or target_input_pos == 'NNS' or target_input_pos == 'NNP' or target_input_pos == 'NNPS':
        return 'noun'
    elif target_input_pos == 'VB' or target_input_pos == 'VBD' or target_input_pos == 'VBG' or target_input_pos == 'VBN' or target_input_pos == 'VBP' or target_input_pos == 'VBZ':
        return 'verb'
    elif target_input_pos == 'JJ' or target_input_pos == 'JJR' or target_input_pos == 'JJS':
        return 'adjective'
    elif target_input_pos == 'RB' or target_input_pos == 'RBR' or target_input_pos == 'RBS':
        return 'adverb'
    else:
        return ''



make_heteronym_list()
