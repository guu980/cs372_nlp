import nltk
import random
from nltk.corpus import *
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def generate_synonyms():
    # 1. Select total 10000 verbs & adjectives from reference corpuses (Gutenberg)
    total_token = []
    # get tokenized words from gutenberg corpuses
    for each_fileid in gutenberg.fileids():
        each_raw_text = gutenberg.raw(each_fileid)
        each_tokenized_text = set(word_tokenize(each_raw_text)) # eliminate duplicated words
        for each_token in each_tokenized_text:
            total_token.append(each_token.lower()) # make all characters to lower case
    print('total words number for reference is %d' % len(total_token)) # for debugging

    total_token_with_tags = pos_tag(total_token) # tag part of speech to words
    verbs_adjs = []
    for each_word in total_token_with_tags:
        if (each_word[0].isalpha()) and (each_word[1] == "VB" or each_word[1] == "JJ"): # select verbs and adjectives among words
            verbs_adjs.append(each_word[0])
    print('total verbs and adjectives number is %d' % len(verbs_adjs))

    random_words = random.sample(verbs_adjs, 10000) # select total 10000 verbs & adjectives randomly
    print('First step is finished. 10000 verbs & adjectives were selected.\n')  # for debugging

    # 2. Find synonyms of the selected words from WordNet
    synonyms = []
    for word in random_words:
        if not wn.synsets(word): # when there are not any synsets for the word (synset is empty)
            continue
        if ( len(wn.synsets(word)[0].lemma_names()) <2 ): # when there is only word slef in synset
            continue
        # From this line, at least synonym list has 2 elements. we have to check
        # 1) synonym should be all alphabet
        # 2) synonym should not be derivative of original word
        # 3) synonym should be verb or adjectives, same as original word
        # we have to check it by converting them into lowercase and campare with original one
        # also we can exclude synset's first word because it is identical with original word
        exclude_first_elem = 0
        for each_synonym in wn.synsets(word)[0].lemma_names():
            exclude_first_elem += 1
            if exclude_first_elem !=1: # exclude synset's first element case
                if each_synonym.isalpha(): # restrict synonym only consist of alphabet (should be single word)
                    if (not each_synonym.lower() in word) and (not
                    word in each_synonym.lower()): # synonym or original word should not be included to the other one
                        if (not each_synonym.lower()[:-3] in word): # synonym should not be derivative. Eliminate suffix by splicing
                            temp_pair = [each_synonym.lower(), word] # pretended common maximum length of suffix is 3
                            temp_tagged_pair = pos_tag(temp_pair) # check part of speech of synony,
                            if (temp_tagged_pair[0][1] == temp_tagged_pair[1][1]) and (temp_tagged_pair[0][1] == "VB" or temp_tagged_pair[0][1] == "JJ"):
                                synonyms.append([word,
                                                 each_synonym.lower(),
                                                 0, 0, temp_tagged_pair[0][1]]) # if all conditions satisfied bring the word
                                break

    print('there are %d synonyms from %d words.' %(len(synonyms), len(random_words))) # for debugging
    print('Second step is finished. Synonyms are extracted by wordnet.\n') # for debugging

    # 3. Compare Frequency for various corpus and calcuate Frequency difference
    # first apply for gutenberg corpuses. Store counts of words and synonym into element's 3rd and 4th field
    for each_fileid in gutenberg.fileids():
        each_text = nltk.Text(gutenberg.words(each_fileid))
        for each_synonym in synonyms:
            each_synonym[2] += each_text.count(each_synonym[0]) # sum original word's frequency
            each_synonym[3] += each_text.count(each_synonym[1]) # sum synonym's frequency

    # second apply for randomly selected several brown corpuses
    for each_fileid in random.sample(brown.fileids(), 70):
        each_text = nltk.Text(brown.words(each_fileid))
        for each_synonym in synonyms:
            each_synonym[2] += each_text.count(each_synonym[0]) # sum original word's frequency
            each_synonym[3] += each_text.count(each_synonym[1]) # sum synonym's frequency

    # third apply for randomly selected several reuters corpuses
    for each_fileid in random.sample(reuters.fileids(), 70):
        each_text = nltk.Text(reuters.words(each_fileid))
        for each_synonym in synonyms:
            each_synonym[2] += each_text.count(each_synonym[0]) # sum original word's frequency
            each_synonym[3] += each_text.count(each_synonym[1]) # sum synonym's frequency

    # calculate the frequency difference and determine word's frequency is wheter bigger or smaller than synonym's
    for each_synonym in synonyms:
        if(each_synonym[2] < each_synonym[3]): # it means A is less frequent than B == A's intensity is stronger than B == intensity modifer should attach on B
            each_synonym[3] = each_synonym[3] - each_synonym[2] # calcuate frequency difference
            each_synonym[2] = 1 # 1 == intensity modifier should attach on synonym(B)
        else: # each_synonym[2] >= each_synonym[3]
            each_synonym[3] = each_synonym[2] - each_synonym[3] # calcuate frequency difference
            each_synonym[2] = 0 # 0 == intensity modifier should attach on word(A)
    print('Third step is finished. Frequencies are calculated.\n') # for debugging

    # 4. Sort list in ascending order by frequency diffeirence and attach intensity modifier to appropriate word
    synonyms.sort(key=lambda synonym: synonym[3]) # sort by frequency differnce
    result=[]
    count = 0
    for each_synonym in synonyms:
        count += 1

        # level 1, difference is smallest == no modifier
        if count < len(synonyms)/5:
            result.append((each_synonym[0], each_synonym[1], each_synonym[4])) # (word(A), synonym(B), theirs part of speech)

        # level 2, difference is becoming bigger == add modifier 'rather'
        elif count < len(synonyms)/5*2:
            if(each_synonym[2] == 0): # when modifier should attach on word(A)
                result.append(('rather ' + each_synonym[0], each_synonym[1], each_synonym[4])) # (rather + word(A), synonym(B), theirs part of speech)
            else: # when modifier should attach on synonym(B)
                result.append(('rather ' + each_synonym[1], each_synonym[0], each_synonym[4])) # (word(A), rather + synonym(B), theirs part of speech)

        # level 3, difference is keep growing == add modifier 'very'
        elif count < len(synonyms)/5*3:
            if(each_synonym[2] == 0): # when modifier should attach on word(A)
                result.append(('very ' + each_synonym[0], each_synonym[1], each_synonym[4])) # (very + word(A), synonym(B), theirs part of speech)
            else: # when modifier should attach on synonym(B)
                result.append(('very ' + each_synonym[1], each_synonym[0], each_synonym[4])) # (word(A), very + synonym(B), theirs part of speech)

        # level 4, difference is large == add modifier 'absolutely'
        elif count < len(synonyms)/5*4:
            if(each_synonym[2] == 0): # when modifier should attach on word(A)
                result.append(('absolutely ' + each_synonym[0], each_synonym[1], each_synonym[4])) # (absolutely + word(A), synonym(B), theirs part of speech)
            else: # when modifier should attach on synonym(B)
                result.append(('absolutely ' + each_synonym[1], each_synonym[0], each_synonym[4])) # (word(A), absolutely + synonym(B), theirs part of speech)

        # level 5, difference is large == add modifier 'extremely'
        else:
            if(each_synonym[2] == 0): # when modifier should attach on word(A)
                result.append(('extremely ' + each_synonym[0], each_synonym[1], each_synonym[4])) # (extremely + word(A), synonym(B), theirs part of speech)
            else: # when modifier should attach on synonym(B)
                result.append(('extremely ' + each_synonym[1], each_synonym[0], each_synonym[4])) # (word(A), extremely + synonym(B), theirs part of speech)
    print('Fourth step is finished. Result words set is created.\n') # for debugging

    # 5. Record result in csv file
    fifty_selected_result = random.sample(result, 50) # exclude duplication
    f = open('result.csv', 'w') # make "result.csv" file
    for each_pair in fifty_selected_result:
        f.write(each_pair[0] + ',' + each_pair[1] + ',' + each_pair[2] + '\n')
    #f.write('\n\n\n') # use it to distinguish with behind results (new intensity modifiers)
    print('Fifth step is finished. Result csv file is created.\n') # for debugging

    # 6. Find new intensity modifier. First collect word which attatched intensity modifier
    modifier_keys = [] # collect in this list
    for each_synonym in synonyms:
        if (each_synonym[2] == 0): # when original word has been attatched
            modifier_keys.append(each_synonym[0])
        else: # when synonym has been attatched
            modifier_keys.append(each_synonym[1])
    print('Sixth step is finished. Modifier keys are created.\n') # for debugging

    # 7. Store the previous word of modifier_key in the corpuses
    modifier_dict = []

    # search gutenberg corpuses
    for each_fileid in gutenberg.fileids():
        each_text_text = nltk.Text(gutenberg.words(each_fileid))
        for each_key in modifier_keys:
            concordance_list_gutenberg = each_text_text.concordance_list(each_key, width=40)
            for each_concline in concordance_list_gutenberg:
                if not each_concline.left: # left is empty
                    continue
                if each_concline.left[-1].isalpha():
                    modifier_dict.append([each_key, each_concline.left[-1].lower()])

    # search randomly selected several brown corpuses
    for each_fileid in random.sample(brown.fileids(), 70):
        each_text_text = nltk.Text(brown.words(each_fileid))
        for each_key in modifier_keys:
            concordance_list_gutenberg = each_text_text.concordance_list(each_key, width=40)
            for each_concline in concordance_list_gutenberg:
                if not each_concline.left: # left is empty
                    continue
                if each_concline.left[-1].isalpha():
                    modifier_dict.append([each_key, each_concline.left[-1].lower()])

    # search randomly selected several reuters corpuses
    for each_fileid in random.sample(reuters.fileids(), 70):
        each_text_text = nltk.Text(reuters.words(each_fileid))
        for each_key in modifier_keys:
            concordance_list_gutenberg = each_text_text.concordance_list(each_key, width=40)
            for each_concline in concordance_list_gutenberg:
                if not each_concline.left: # left is empty
                    continue
                if each_concline.left[-1].isalpha():
                    modifier_dict.append([each_key, each_concline.left[-1].lower()])
    print('Seventh step is finished. Modifier dictionary is created with %d elements.\n' %len(modifier_dict)) # for debugging

    # 8. Attach tag of part of speech and extract only previous word was adverb
    modifier_result = []
    for each_kv in modifier_dict:
        modifier_tagged_dict = pos_tag(each_kv)
        if(modifier_tagged_dict[1][1] == "RB"): # when previous word is adverb
            modifier_result.append([modifier_tagged_dict[1][0], modifier_tagged_dict[0][0], 0])
    print('Eighth step is finished. Modifier result is created with %d elements.\n' % len(modifier_result)) # for debugging

    # 9. Look for how many phrase (previous word + (word or synonym)) is in gutenberg corpuses
    for each_fileid in gutenberg.fileids():
        each_text = gutenberg.raw(each_fileid)
        for each_elem in modifier_result:
            each_elem[2] += each_text.count(each_elem[0] + " " + each_elem[1]) # sum the total count of phrase

    for each_fileid in random.sample(brown.fileids(), 70):
        each_text = brown.raw(each_fileid)
        for each_elem in modifier_result:
            each_elem[2] += each_text.count(each_elem[0] + " " + each_elem[1]) # sum the total count of phrase

    for each_fileid in random.sample(reuters.fileids(), 70):
        each_text = reuters.raw(each_fileid)
        for each_elem in modifier_result:
            each_elem[2] += each_text.count(each_elem[0] + " " + each_elem[1]) # sum the total count of phrase

    estimated_modifiers = []
    for each_elem in modifier_result:
        if (each_elem[2] > 4): # if phrase appeared more than 4 times
            estimated_modifiers.append(each_elem[0]) # consiedr it would be intensity modifier nad include it into result
    estimated_modifiers = set(estimated_modifiers) # eliminate duplication
    print('Nineth step is finished. There are %d modifiers.\n' % len(estimated_modifiers)) # for debugging
    for each_elem in estimated_modifiers:
        f.write(each_elem + '\n') # record the newly produced intensity modifying adverb

generate_synonyms()
