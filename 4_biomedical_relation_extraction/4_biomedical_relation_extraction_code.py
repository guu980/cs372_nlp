import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from pprint import pprint
from nltk.tree import ParentedTree

# Main function which shows big pipeline
def main():

    # Load randomly selected 80 training sentences from excel file
    # Used pandas to avoid csv's syntax breaking because many data include ',' inside them so it is hard to distinguish them with indicator role's ',' of csv
    train_xlsx = pd.read_excel('./input_80_sent.xlsx')
    input_train_data = train_xlsx.values.tolist()

    # Extract actual answer triples from loaded training data
    '''
    -------IMPORTANT-------
    In excel file, I used @@@ to distinguish x,action,y data from other triples since if we use ',' then we can distinguish
    wheter it is x,action,y's string data include ',' or it is splitor for different triples
    -------IMPORTANT-------
    '''
    actual_train_triples = []
    for each_row in input_train_data:
        xs = each_row[0]
        xs = xs.split('@@@')
        actions = each_row[1]
        actions = actions.split('@@@')
        ys = each_row[2]
        ys = ys.split('@@@')

        each_row_actual_triples = []
        for i in range(0,len(xs)):
            each_row_actual_triples.append((xs[i], actions[i], ys[i]))
        actual_train_triples.append(each_row_actual_triples)

    # Apply module to produce system triples
    system_train_triples = []
    for each_row in input_train_data:
        sentence = each_row[3]
        system_train_triples.append(get_sentence_triple(sentence))

    # Evaluate the performance for training data
    calculated_precision, calculated_recall, calculated_f_score = evaluate(actual_train_triples, system_train_triples)
    print('Evaluation completed for 80 sentences. Precision is {}, recall is {}, f_score is {}'.format(calculated_precision, calculated_recall, calculated_f_score))

    # Load randomly selected 20 test sentences form escel file. Other detailed condition is same with above explanation
    test_xlsx = pd.read_excel('./input_20_sent.xlsx')
    input_test_data = test_xlsx.values.tolist()

    # Extract actual answer triples from loaded test data
    actual_test_triples = []
    for each_row in input_test_data:
        xs = each_row[0]
        xs = xs.split('@@@')
        actions = each_row[1]
        actions = actions.split('@@@')
        ys = each_row[2]
        ys = ys.split('@@@')

        each_row_actual_triples = []
        for i in range(0,len(xs)):
            each_row_actual_triples.append((xs[i], actions[i], ys[i]))
        actual_test_triples.append(each_row_actual_triples)

    # Apply module to produce system triples
    system_test_triples = []
    for each_row in input_test_data:
        sentence = each_row[3]
        system_test_triples.append(get_sentence_triple(sentence))

    # Evaluate the performance for test data
    calculated_precision, calculated_recall, calculated_f_score = evaluate(actual_test_triples, system_test_triples)
    print('Evaluation completed for 20 sentences. Precision is {}, recall is {}, f_score is {}'.format(calculated_precision, calculated_recall, calculated_f_score))
    print()

    # Print out result in csv file
    '''
    --------IMPORTANT--------
    During process when I write data into csv file, I found excel or hancel program doesn't show the result well since
    comma (,) in the sentence break the syntax of csv file. So it seems like sentence number is smaller than 100.
    I want to clarify there are "100" sentences's data in real, and recorded in csv file, but reason data's row number
    is smaller than 100 is because of comma csv syntax break. I checked there are no problems when I open in text editor
    and if I remove code's 121, 122, 166, 167 lines which records sentence information into the csv file, 100 rows are
    perfectly shown in excel. Please check it if you want to check by yourself
    --------IMPORTANT--------
    '''
    f = open('result.csv', 'w', -1, 'utf-8') # make "result.csv" file
    f.write('x,action,y,sentence,PMID,year,journal title,organization' + '\n') # for categorization in first row

    # Record module applied result for training data. Same as above, I used @@@ to split data of different triples
    for i in range(0, len(input_train_data)):
        writing_x = ''
        writing_action = ''
        writing_y = ''

        # Record triples information x,action,y
        if len(system_train_triples[i])>0:
            for each_row_triple in system_train_triples[i]:
                if writing_x == '':
                    writing_x = writing_x + each_row_triple[0]
                else:
                    writing_x = writing_x + "@@@" + each_row_triple[0]

                if writing_action == '':
                    writing_action = writing_action + each_row_triple[1]
                else:
                    writing_action = writing_action + "@@@" + each_row_triple[1]

                if writing_y == '':
                    writing_y = writing_y + each_row_triple[2]
                else:
                    writing_y = writing_y + "@@@" + each_row_triple[2]

        f.write(writing_x)
        f.write(',')
        f.write(writing_action)
        f.write(',')
        f.write(writing_y)
        f.write(',')

        f.write(input_train_data[i][3]) # record sentence remove this part if you want to check data's number
        f.write(',') # remove this part if you want to check data's number
        f.write(str(input_train_data[i][4])) # record PMID
        f.write(',')
        f.write(str(input_train_data[i][5])) # record year
        f.write(',')
        f.write(input_train_data[i][6]) # record journal title
        f.write(',')
        f.write(str(input_train_data[i][7])) # record organization
        f.write(',')
        f.write('\n')

    f.write('\n')

    # Record module applied result for test data. Same as above, I used @@@ to split data of different triples
    for i in range(0, len(input_test_data)):
        writing_x = ''
        writing_action = ''
        writing_y = ''

        # Record triples information x,action,y
        if len(system_train_triples[i])>0:
            for each_row_triple in system_train_triples[i]:
                if writing_x == '':
                    writing_x = writing_x + each_row_triple[0]
                else:
                    writing_x = writing_x + "@@@" + each_row_triple[0]

                if writing_action == '':
                    writing_action = writing_action + each_row_triple[1]
                else:
                    writing_action = writing_action + "@@@" + each_row_triple[1]

                if writing_y == '':
                    writing_y = writing_y + each_row_triple[2]
                else:
                    writing_y = writing_y + "@@@" + each_row_triple[2]

        f.write(writing_x)
        f.write(',')
        f.write(writing_action)
        f.write(',')
        f.write(writing_y)
        f.write(',')

        f.write(input_test_data[i][3]) # record sentence remove this part if you want to check data's number
        f.write(',') # remove this part if you want to check data's number
        f.write(str(input_test_data[i][4])) # record PMID
        f.write(',')
        f.write(str(input_test_data[i][5])) # record year
        f.write(',')
        f.write(input_test_data[i][6]) # record journal title
        f.write(',')
        f.write(str(input_test_data[i][7])) # record organization
        f.write(',')
        f.write('\n')

    print('Printing results is finished \n')  # for debugging
    return


# Function that extract triples from each sentence
def get_sentence_triple(sentence):
    # Auxiliary functions
    # Function that extract triples from NLTK tree
    def get_triples(parent):
        Triples = []

        for node_idx, node in enumerate(parent):
            if type(node) is nltk.tree.ParentedTree:
                if node.label() == 'ACTION':
                    x, y = findxy(node_idx, node.parent(), True, True)
                    action = node.leaves()[0][0]

                    # find action's passive form
                    be_verb = ['is', 'are', 'was', 'were']
                    if (type(parent[node_idx-1]) is not nltk.tree.ParentedTree) and (parent[node_idx-1][0] in be_verb):
                        if (type(parent[node_idx+1]) is not nltk.tree.ParentedTree) and (parent[node_idx+1][1] == "IN"):
                            action = parent[node_idx-1][0] + " " + action + " " + parent[node_idx+1][0]

                    # only consider when x and y is properly produced (exclude case like gerund form or pronoun etc)
                    if x!=None and y!=None:
                        Triples.append((x, action, y))

        return Triples

    # Function that find x and y from phrase
    def findxy(action_idx, phrase_node, search_x, search_y):
        x_candidates = []
        y_candidates = []

        for node_idx, node in enumerate(phrase_node):
            if type(node) is nltk.tree.ParentedTree:
                # Search last and first NP from ACTION to find x,y
                if node.label() == 'NP':
                    if search_x:
                        if node_idx < action_idx:
                            x_candidates.append(extract_cons_n_from_np(node))

                    if search_y:
                        if node_idx > action_idx:
                            y_candidates.append(extract_cons_n_from_np(node))
                            break

        if len(x_candidates)==0:
            x = None
        else:
            x = x_candidates[-1]

        if len(y_candidates)==0:
            y = None
        else:
            y = y_candidates[0]

        return (x, y)

    # Function that extract consecutive nouns from np node
    def extract_cons_n_from_np(np_node):
        label_list = [each_node.label() for each_node in np_node if type(each_node) is nltk.tree.ParentedTree]

        # if there is CLOSE_DJNS, use it since it has possibility of named entity
        if 'CLOSE_DJNS' in label_list:
            close_djns_list = []
            for each_node in np_node:
                if each_node.label() == 'CLOSE_DJNS':
                    close_djns_list.append(each_node)

            for each_node_in_close_djns in close_djns_list[-1]:
                if type(each_node_in_close_djns) is nltk.tree.ParentedTree:
                    if each_node_in_close_djns.label() == 'CONS_N':
                        result = ''
                        for each_noun in each_node_in_close_djns.leaves():
                            # exclude proper noun case
                            if each_noun[1] == 'PRP':
                                return None

                            if result != '':
                                result += " " + each_noun[0]
                            else:
                                result += each_noun[0]
                        return result

        # if there is no CLOSE_DJNS use last DJNS
        else:
            for each_node_in_djns in np_node[-1]:
                if type(each_node_in_djns) is nltk.tree.ParentedTree:
                    if each_node_in_djns.label() == 'CONS_N':
                        result = ''
                        for each_noun in each_node_in_djns.leaves():
                            # exclude proper noun case
                            if each_noun[1] == 'PRP':
                                return None

                            if result != '':
                                result += " " + each_noun[0]
                            else:
                                result += each_noun[0]
                        return result

    # Pipeline start point

    # 1. Tag the POS, find action and change it's pos to ACT in the sentence
    pos_tagged_tokens = pos_tag(word_tokenize(sentence))

    action_list = ['activate', 'inhibit', 'bind', 'accelerate', 'augment', 'induce', 'stimulate', 'require',
                   'up-regulate', 'abolish', 'block', 'down-regulate', 'down-regulated', 'prevent']
    wnl = nltk.WordNetLemmatizer()
    for index,value in enumerate(pos_tagged_tokens):
        word = value[0]
        pos = value[1]
        if wnl.lemmatize(word.lower(), 'v') in action_list:
            # exclude gerund form
            if pos != 'VBG':
                pos_tagged_tokens[index] = (word, 'ACT')

    # 2. Write own grammar
    grammar = r"""
    CONS_N: 
        {<NN.*>+}
        {<PRP>}
        
    CLOSE_DJNS: {<\(> <DT|P\$>? <JJ>* <CONS_N> <\)>}
    
    DJNS: {<DT|PRP\$>? <JJ>* <CONS_N>}
    
    NP: 
        {<DJNS>+ <CLOSE_DJNS>+ <DJNS>*}
        {<CLOSE_DJNS>+ <DJNS>*}
        {<DJNS>+}

    ACTION: {<ACT>}
    
    IN_COMMA_BEF_ACTION:
        {<,> <[^,][^,][^,]?>+ <,> <MD>? <RB.?>* <ACTION>}
        } <MD>? <RB.?>* <ACTION> {
        
    TO_BEF_ACTION:
        {<TO> <RB.?>* <ACTION>}
    """

    # 3. Chunking by grammar
    cp = nltk.RegexpParser(grammar, loop=3)
    cs = cp.parse(pos_tagged_tokens)
    parented_cs = ParentedTree.convert(cs)
    triples = get_triples(parented_cs)

    return triples


# Function that assess the performance. Calculate precision, recall, f_score
def evaluate(actual_triples, system_triples):
    total_system_triples_num = 0
    for each_row_triples in system_triples:
        total_system_triples_num += len(each_row_triples)

    total_actual_triples_num = 0
    for each_row_triples in actual_triples:
        total_actual_triples_num += len(each_row_triples)

    tp_triples_num = 0
    for i in range(0, len(actual_triples)):
        for each_row_system_triple in system_triples[i]:
            for each_row_actual_triple in actual_triples[i]:
                system_x = each_row_system_triple[0]
                system_action = each_row_system_triple[1]
                system_y = each_row_system_triple[2]

                actual_x = each_row_actual_triple[0]
                actual_action = each_row_actual_triple[1]
                actual_y = each_row_actual_triple[2]

                if (system_x in actual_x) and system_action == actual_action and (system_y in actual_y):
                    tp_triples_num += 1
                    break

    precision = tp_triples_num / total_system_triples_num
    recall = tp_triples_num / total_actual_triples_num
    f_score = (2 * precision * recall) / (precision + recall)
    return (precision, recall, f_score)



main()