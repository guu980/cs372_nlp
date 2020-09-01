import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from pprint import pprint

import spacy
from spacy import displacy
from nltk.corpus import names

import requests
from bs4 import BeautifulSoup
import pickle
import os

import csv

"""
Function to count start point's offset of each sentence
return the sentence length counted by charactes
"""
def count_sent_chars_length(sent):
    sent_length = 0
    for each_word in sent:
        sent_length += len(each_word)
    return sent_length



"""
Function to crawl title and summary data from widipedia URL
return the tuple of title text data and summary text dat
"""
def crawl_wiki(row):
    title_text = ""
    summary_text = ""

    # Get HTML file
    try:
        response = requests.get(row[10])

    except:
        # There were some connection failure cases when I executed in my home
        # Reporting and debugging for those cases
        print("Connection failed\n")
        return

    # Get title data
    soup = BeautifulSoup(response.content, "html.parser")
    head_info = soup.find('h1', class_="firstHeading")
    title_text = head_info.get_text()
    print("Title: " + title_text) # for debugging

    # Get summary data
    main_info = soup.find('div', class_="mw-parser-output")
    if main_info != None:
        main_info = main_info.findAll('p')
        if main_info != None:
            # Summary data was usually second p
            try:
                summary_text = main_info[1].get_text()
                if summary_text == '\n' or summary_text == '':
                    summary_text = main_info[0].get_text()
                    if summary_text == '\n' or summary_text == '':
                        summary_text = main_info[2].get_text()

                print("Summary: " + summary_text)

            # Hande cases when we can't find second p
            except IndexError:
                print("Summary: ")
                print(main_info)
                print("There are no elements more than two")

        # It can be None when there is no documentation in wikipedia
        else:
            print("None case 2\n")

    # It can be None when there is no documentation in wikipedia
    else:
        print("None case 1\n")

    return (title_text, summary_text)



"""
Function to check pronoun's gender property and A or B's gender property match with pronoun's gender property or not.
If gender property matches, allocate True, if doesn't match, allocate Faalse
return "TRUE" if gender property matches, "FALSE" when it doesnt match
"""
def is_satisfy_gender(male_names, female_names, idx, pronoun, input):

    def check_each_input(man, input):
        # Male case. Check whether name is male's or not
        if man:
            if input in male_names:
                return True
            else:
                return False

        # Female case. Check whether name is female's or not
        else:
            if input in female_names:
                return True
            else:
                return False

    male_pronouns = ['he', 'his', 'him']
    female_pronouns = ['she', 'her', 'hers']
    input_satisfy_gender = False

    if pronoun in male_pronouns:
        # Case when pronoun's gender property is male
        input_satisfy_gender = check_each_input(True, input)

    elif pronoun in female_pronouns:
        # Case when pronoun's gender property is female
        input_satisfy_gender = check_each_input(False, input)

    else:
        print("There is no corresponding pronouns in pre-captured list!!! idx is {}".format(idx)) # for debugging

    return input_satisfy_gender



"""
Function to handle gender corner case. The case when right value is "TRUE", but is_satisfy_gender() result is "FALSE".
This function is only used once manually by me during the making gender checking system.
Therefore, it doesnt' directly be used in main() function. I left it to show how it works
Please check the report if you want to know more in detail.
As result, it produces male_names.txt and female_names.txt files which have trained name data based on names nltk corpus
"""
def handle_gender_conrner_case():

    male_names = names.words('male.txt')
    female_names = names.words('female.txt')

    male_pronouns = ['he', 'his', 'him']
    female_pronouns = ['she', 'her', 'hers']

    # Prepare to read and write the file
    f = open('gap-test.tsv', 'r', encoding='utf-8')
    input_reader = csv.reader(f, delimiter='\t')
    snippet_result_f = open('snippet_result.tsv', 'r', encoding='utf-8', newline='')
    snippet_result_reader = csv.reader(snippet_result_f, delimiter='\t')

    input_data = []
    snippet_result_data = []

    # Read answer (gold) data
    for idx, row in enumerate(input_reader):
        if idx > 0:
            input_data.append((row[0], row[6], row[9], row[2], row[4], row[7]))

    # Read first result data of gender checking
    for idx, row in enumerate(snippet_result_reader):
        snippet_result_data.append((row[0], row[1], row[2]))

    # Training
    for i in range(0,2000):
        if input_data[i][0] != snippet_result_data[i][0]:
            print("Something's wrong. ID doesn't match. Input's id: {}, snippet's ids: {}".format(input_data[i][0], snippet_result_data[i][0])) # for debugging
        else:
            if input_data[i][1] == "TRUE" and snippet_result_data[i][1] == "FALSE":
                if input_data[i][3].lower() in male_pronouns:
                    # Male case
                    male_names.append(input_data[i][4])
                elif input_data[i][3].lower() in female_pronouns:
                    # Female case
                    female_names.append(input_data[i][4])
                else:
                    print("Somethings wrong. Pronoun is not included in male or female. This word is {}".format(input_data[i][3].lower())) # for debugging

            if input_data[i][2] == "TRUE" and snippet_result_data[i][2] == "FALSE":
                if input_data[i][3].lower() in male_pronouns:
                    # Male case
                    male_names.append(input_data[i][5])
                elif input_data[i][3].lower() in female_pronouns:
                    # Female case
                    female_names.append(input_data[i][5])
                else:
                    print("Somethings wrong. Pronoun is not included in male or female. This word is {}".format(input_data[i][3].lower())) # for debugging

    # Save male and female names list into txt file
    with open('male_names.txt', 'wb') as mn_f:
        pickle.dump(male_names, mn_f)
    with open('female_names.txt', 'wb') as fmn_f:
        pickle.dump(female_names, fmn_f)



"""
Function to conduct coreference resolution in snippet context.
Needs 4 parameters. Male names list, female naems list, each index and row data of gap-test.tsv

row[0] = ID, row[1] = Text, row[2] = Pronoun, row[3] = Pronoun offset
row[4] = A, row[5] = A-offset, row[6] = A-coreference,
row[7] = B, row[8] = B-offset, row[9] = B-coreference
row[10] = URL
"""
def analyze_coreference_snippet(male_names, female_names, idx, row):

    a_coref = "FALSE"
    b_coref = "FALSE"

    # 1. Filtering with gender
    word_tokenized_a = word_tokenize(row[4])
    word_tokenized_b = word_tokenize(row[7])

    # Check A first
    for word in word_tokenized_a:
        a_satisfy_gender = is_satisfy_gender(male_names, female_names, idx, row[2].lower(), word)
        if a_satisfy_gender:
            a_coref = "TRUE"
            break

    # Check B next
    for word in word_tokenized_b:
        b_satisfy_gender = is_satisfy_gender(male_names, female_names, idx, row[2].lower(), word)
        if b_satisfy_gender:
            b_coref = "TRUE"
            break

    # 2. Filtering with syntax for the cases A and B is both True in gender checking process
    if a_coref == "TRUE" and b_coref == "TRUE":
        sent_tokenized_text = sent_tokenize(row[1])
        pronoun_including_sent_list = []

        # Calculate starting offset of each senteces and find sentences which include target pronoun
        cursor = 0
        for sent_idx, each_sentence in enumerate(sent_tokenized_text):
            if row[2] in each_sentence:
                pronoun_including_sent_list.append((sent_idx, each_sentence, cursor))
                cursor += count_sent_chars_length(each_sentence)

        # Find real target pronoun including sentence based on offset
        if len(pronoun_including_sent_list) > 1 or len(pronoun_including_sent_list) == 0:
            for each_sent_elem_idx, each_sent_elem in enumerate(pronoun_including_sent_list):
                try:
                    if each_sent_elem[2] <= int(row[3]) and pronoun_including_sent_list[each_sent_elem_idx + 1][2] > int(row[3]):
                        pronoun_including_sent_list = [each_sent_elem]
                        break
                except:
                    pronoun_including_sent_list = [pronoun_including_sent_list[-1]]
                    break

            if len(pronoun_including_sent_list) != 1:
                print("Still pronoun including sentence's number is {}! Somethings wrong".format(len(pronoun_including_sent_list))) # for debugging
                return (False, False)

        # Determine A should be False or B should be False
        pronoun_including_sent_idx = pronoun_including_sent_list[0][0]
        pronoun_including_sent = pronoun_including_sent_list[0][1]

        # Case when pronoun including sentence is first sentence of the input text
        if pronoun_including_sent_idx == 0:
            if int(row[5]) > int(row[8]):
                b_coref = "FALSE"
            else:
                a_coref = "FALSE"

        # Case when pronoun including sentence is not first sentence of the input text
        else:
            # Dependency parsing
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(pronoun_including_sent)
            dep_parsed_sent = []
            for token_idx, token in enumerate(doc):
                dep_parsed_sent.append((token_idx, token.text, token.dep_))

            # Find root and pronoun position and compare them
            root_idx = 0
            prop_idx = 0
            for each_token in dep_parsed_sent:
                if each_token[2] == "ROOT":
                    root_idx = each_token[0]
                if each_token[1] == row[2]:
                    prop_idx = each_token[0]
            if root_idx < prop_idx:
                # Pronoun after root
                if int(row[5]) > int(row[8]):
                    b_coref = "FALSE"
                else:
                    a_coref = "FALSE"
            else:
                # Pronoun before root
                if int(row[5]) > int(row[8]):
                    a_coref = "FALSE"
                else:
                    b_coref = "FALSE"

    return (a_coref, b_coref)



"""
Function to conduct coreference resolution in page context.
Needs 3 parameters. Wikipedia data list, each index and row data of gap-test.tsv

row[0] = ID, row[1] = Text, row[2] = Pronoun, row[3] = Pronoun offset
row[4] = A, row[5] = A-offset, row[6] = A-coreference,
row[7] = B, row[8] = B-offset, row[9] = B-coreference
row[10] = URL
"""
def analyze_coreference_page(wiki_data, idx, row):

    a_coref = "FALSE"
    b_coref = "FALSE"

    # Check A appears in title or summary of Wikipedia documentation
    if row[4] in wiki_data[idx][0] or row[4] in wiki_data[idx][1]:
        a_coref = "TRUE"

    # Check B appears in title or summary of Wikipedia documentation
    if row[7] in wiki_data[idx][0] or row[7] in wiki_data[idx][1]:
        b_coref = "TRUE"

    return (a_coref, b_coref)



"""
Main function to conduct coreference resolution and make result files in tsv format.
As a result, it produces page_result.tsv, and snippet_result.tsv gap-system-output format.
"""
def main():

    # Prepare files
    f = open('gap-test.tsv', 'r', encoding='utf-8')
    input_reader = csv.reader(f, delimiter='\t')
    snippet_result_f = open('snippet_result.tsv', 'w', encoding='utf-8', newline='')
    snippet_result_writer = csv.writer(snippet_result_f, delimiter='\t')
    page_result_f = open('page_result.tsv', 'w', encoding='utf-8', newline='')
    page_result_writer = csv.writer(page_result_f, delimiter='\t')

    # Prepare Wikipidea data
    wiki_data = dict()

    # If wikipedia data already exists in file, use it
    if os.path.isfile('wiki_data.txt'):
        with open('wiki_data.txt', 'rb') as wiki_f:
            wiki_data = pickle.load(wiki_f)
        print("Wiki data exists. Loaded data\n") # for debugging

    # If wikipedia data doesn't exists in file, crawl the data
    else:
        print("There is no wikipidea data. Start crawling") # for debugging

        for idx, row in enumerate(input_reader):
            if idx > 0:
                print(idx)
                wiki_data[idx] = crawl_wiki(row)

        # Save the produced data into file
        with open('wiki_data.txt', 'wb') as wiki_f:
            pickle.dump(wiki_data, wiki_f)

        print("Crawling finished. Made wiki data") # for debugging

    # Prepare male and female names list data
    male_names = []
    female_names = []

    # Male data
    # If gender names list data already exists in file, use it
    if os.path.isfile('male_names.txt'):
        with open('male_names.txt', 'rb') as mn_f:
            male_names = pickle.load(mn_f)
        print("Male names data exists. Loaded data") # for debugging

    # If gender names list data doesn't exist in file, create it based on nltk names corpus
    else:
        male_names = names.words('male.txt')
        # Save the produced data into file
        with open('male_names.txt', 'wb') as mn_f:
            pickle.dump(male_names, mn_f)
        print("Created male names list") # for debugging

    # Female data
    # If gender names list data already exists in file, use it
    if os.path.isfile('female_names.txt'):
        with open('female_names.txt', 'rb') as fmn_f:
            female_names = pickle.load(fmn_f)
        print("Female names data exists. Loaded data") # for debugging

    # If gender names list data doesn't exist in file, create it based on nltk names corpus
    else:
        female_names = names.words('female.txt')
        # Save the produced data into file
        with open('female_names.txt', 'wb') as fmn_f:
            pickle.dump(female_names, fmn_f)
        print("Created female names list") # for debugging

    # Coreference resolution
    print("Start analyzing corefernce") # for debugging
    for idx,row in enumerate(input_reader):
        print(idx) # for debugging

        if idx > 0:
            # Snippet context coreference resolution
            snippet_a_coref, snippet_b_coref = analyze_coreference_snippet(male_names, female_names, idx, row)
            snippet_result_writer.writerow([row[0], snippet_a_coref, snippet_b_coref])

            # Page context coreference resolution
            page_a_coref, page_b_coref = analyze_coreference_page(wiki_data, idx, row)
            page_result_writer.writerow([row[0], page_a_coref, page_b_coref])

    print("Analyzing finish") # for debugging
    f.close()
    page_result_f.close()

    return



"""
Functions to turn wiki_data.txt files into wiki_data.csv files.
wiki_data.csv files doesn't be used directly in main function, but I used it to examine Wikipedia crawled data
and get some intuition from them to implement page context function
As a result, it produces wiki_data.csv
"""
def wiki_txt_to_csv():

    # Prepare files
    wiki_f = open('wiki_data.csv', 'w', encoding='utf-8', newline='')
    wiki_writer = csv.writer(wiki_f)
    wiki_writer.writerow(['ID', "Title", "Summary"])
    wiki_data_txt = open('wiki_data.txt', 'rb')
    wiki_data = pickle.load(wiki_data_txt)

    # Converting process
    for i in range(1, 2001):
        wiki_writer.writerow([i, wiki_data[i][0], wiki_data[i][1]])

    return



main()