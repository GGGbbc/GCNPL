
"""
Define constants for semeval-10 task.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]


POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46, 'pad': 47}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

NEGATIVE_LABEL = 'None'
# hard-coded mappings from fields to ids
#-------------------------------------cpr--------------------------------
# SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'CHEMICAL': 2}
# OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'CHEMICAL': 2, 'GENE-Y': 3,'GENE-N': 4}
# NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'CHEMICAL': 3, 'GENE-Y': 4,'GENE-N': 5}
#LABEL_TO_ID = {'CPR:3': 0, 'CPR:4': 1, 'CPR:5': 2, 'CPR:6': 3, 'CPR:9': 4, 'None': 5}

#-------------------------------------pgr--------------------------------
# SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PHENOTYPE': 2, 'GENE': 3}
# OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PHENOTYPE': 2, 'GENE': 3}
# NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PHENOTYPE': 3, 'GENE': 4,}
# LABEL_TO_ID = {'True': 0, 'False': 1}

#-------------------------------------ddi--------------------------------
# SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'DRUG1': 2, 'DRUG2': 3}
# OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'DRUG1': 2, 'DRUG2': 3}
# NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'DRUG1': 3, 'DRUG2': 4,}
# LABEL_TO_ID = {'DDI-mechanism': 0, 'DDI-advise': 1, 'DDI-effect': 2, 'DDI-int': 3, 'None': 4}


#-------------------------------------gad--------------------------------
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'GENE': 2, 'DISEASE': 3}
OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'GENE': 2, 'DISEASE': 3}
NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'GENE': 3, 'DISEASE': 4,}
LABEL_TO_ID = {'True': 0, 'False': 1}



INFINITY_NUMBER = 1e12



































