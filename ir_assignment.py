import re                                   #Regular Expression
import nltk                                 #For Lemmatizationa and stopwords removal
import mxnet as mx                          #For interfacing cuda 
import numpy as np                          #Array Data Structure and Cosine Score
from datasets import load_dataset           #load_dataset function form 'dataset' library to load dataset
from nltk.corpus import stopwords           #Get stopwords
from bert_embedding import BertEmbedding    #BertEmbeddings function gives embeddings
from nltk.stem import WordNetLemmatizer 

nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;!@]')

def preprocess(text):
    """
    string: a string\n
    return: preprocessed string\n
    Takes text as input and \n
        1) Makes everything lower case.\n
        2) Deletes '[/(){}\[\]\|@,;!@]' and then '[^0-9a-z #+_]'.\n
        3) Removes StopWords.\n
        4) Lemmatizes every word.\n
    """
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(" " , text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub("", text)         # delete symbols which are in BAD_SYMBOLS_RE from text
    textlist = text.split()
    text = [wrd for wrd in textlist if wrd not in STOPWORDS]
    text = " ".join(text)
    
    #Lemmatize
    lem_words = []
    for word in text.split():
        lem_words.append(lemmatizer.lemmatize(word))
    return " ".join(lem_words)

def load_dataset_and_preprocess(emails):
    """
    Input:None\n
    Load dataset and preprocess it\n
    Output:Dataset and preprocessed Dataset\n
    """
    #unpreprocessed
    email_sent = []
    """Array of sentences in the emails"""
    for i in range(len(emails)):
        email_sent.append([i for i in emails[i].split('\n') if len(i)!=0 ])

    #preprocessed
    corpus = []
    """Hold the Preprocessed Corpus"""
    for email in emails:
        email = email.split("\n")
        preprocessed_sentence=[]
        for i in range(len(email)):
            if len(email[i]) > 0:
                prep_s = preprocess(email[i])
                if len(prep_s) > 0:
                    preprocessed_sentence.append(prep_s)
        corpus.append(preprocessed_sentence)
    return email_sent, corpus

"""# BERT"""

def build_bert_embeddings_index(corpus):
    """
    Input: Dataset as corpus\n
    Build BERT Embeddings of sentences averaged over words in each sentence and index them\n
    Output: Embeddings of each sentence in Dataset\n
    """
    try:
        ctx = mx.gpu(0)
        """Set context to GPU on Mxnet"""
        be = BertEmbedding(ctx=ctx)
        """Function to get embeddings of each word in array of sentences """
    except:
        be = BertEmbedding()
    embeddings = []
    """Store all the Sentence Embeddings in the corpus"""
    for email in range(len(corpus)):
        printc=True
        for i in be(corpus[email]):
            if email%500==0 and printc:
                print("Iteration: ", email)
                printc=False
            try:
                embeddings.append(sum(i[1])/len(i[1]))
            except:
                print(i, email, corpus[email])
    return embeddings

def build_reference(corpus):
    """
    Input: Dataset as corpus\n
    Build reference of each line to email and starting line of the email.\n
    Output: Reference index from line to email \n
    """
    indx_to_email = dict()
    """Dictionary to refer the corpus line number (l) to the Email number and the fist line number of the email"""
    l=0
    for indx_email in range(len(corpus)):
        start = l
        for j in range(len(corpus[indx_email])):
            indx_to_email[l]=(start, indx_email)
            l+=1
    return indx_to_email

def find_email(i_email, unprep, highlight_line=None):
    """
    Input:Index of email and the email dataset, optional line number to highlight\n
    Prints the email with optional highlighted line\n
    Output: Prints the same\n
    """
    for i in range(len(unprep[i_email])):
        if i==highlight_line:
            print('\33[34m' + unprep[i_email][i] + '\033[0m')
        else:
            print(unprep[i_email][i])

def cosine_score(vec1, vec2):
    """
    Input: Two embeddings of same dimentions\n
    Finds the cosine score among them\n
    Output: Cosine Similarity\n
    """
    try:
        if vec1.all() == None or vec2.all() == None:
            return 0
    except: 
        return 0
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

def bert_search(query, embeddings, corpus, indx_to_email, top_n=3, is_preprocess=False):  
    """
    Input:Index of email and the email dataset, optional line number to highlight\n
    Prints the email with optional highlighted line\n
    Output: Prints the same\n
    """  
    try:
        ctx = mx.gpu(0)
        be = BertEmbedding(ctx=ctx)
    except:
        be = BertEmbedding()

    l = 0
    q = query

    c = [i for i in q.split('\n') if len(i)!=0]
    """Split query per line and then """

    if is_preprocess:
        que = [preprocess(i) for i in q.split('\n') if len(i)!=0 ] 
    else: 
        que = q.split('\n')
    
    embedding = be(que)
    """Array to store the splitted words in the query and their corresponding embeddings"""
    for i in range(len(embedding)):
        print("\33[35mQuery {} of {} \033[0m".format(l+1, len(embedding)))
        e = embedding[i][1]
        """Array of embeddings of each word in the sentence"""
        sent_embedding = sum(e)/len(e)
        """Averaged embedding of the sentence"""
        similarity = []
        """Array to store the similarity with embeddings of each sentence in the corpus"""
        for j in embeddings:
            similarity.append(cosine_score(sent_embedding, j))

        sim = np.argsort(similarity)[::-1][:top_n]
        """Array to store the top n similarities"""
        print("Query:", "\33[31m" + c[l] + "\033[0m")
        if is_preprocess:
            print("Preprocessed query:", "\33[31m" + que[i]+ "\033[0m" "\n")
        l+=1
        for i in sim:
            print("\33[33m" + "Cosine Similarity: {:.3f} \t Email {} \t Line {} \033[0m".format(similarity[i], indx_to_email[i][1], i-indx_to_email[i][0]))
            print("\33[34m" + str(corpus[indx_to_email[i][1]][i-indx_to_email[i][0]]) + '\033[0m',  "\n")

def set_dataset(dataset):
    return dataset

def init_glove_embeddings(dimentions=300):
    """
    Input: Dimentions of the Embeddings to use\n
    Builds the glove embeddings in a dicyionary from the file\n
    Output: Dictionary of the pretrained dictionary\n
    """
    if dimentions not in [50, 100, 200, 300]:
        print("Embeddings only Exists for (50, 100, 200, 300) dimentions")
        return "Embeddings for selected dimensions doesn't exist"
    glove_embeddings = {}
    """Dictionary to store the embeddings from Glove file"""
    with open("glove.6B."+str(dimentions)+"d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_embeddings[word] = vector
    return glove_embeddings

def glove_embed(sent):
    """
    Input: Sentence as string\n
    Finds the glove embeddings of each word and adds it for each sentence\n
    Output: Embeddings and the length of words taken to form the embeddings\n
    """
    sent = sent.lower()
    embeddings = [0 for _ in embeddings_dict["king"]]
    """Array to store query embeddings initialized to 0"""
    length = 0
    for word in sent.split():
        try:
            embeddings+=embeddings_dict[word]
            length+=1
        except:
            pass
            
    return embeddings, length

def build_glove_index(corpus):
    """
    Input: Corpus\n
    Builds the index of embeddings of each sentence in the corpus\n
    Output: Embddeings Index of each sentence in the corpus\n
    """
    embeddings = []
    """Array to store the sentence Embeddings of the dataset"""
    sentence_corpus = []
    """Array of strings to store the sentence in the corpus"""
    for email in range(len(corpus)):
        printc=True
        for i in corpus[email]:
            sentence_corpus.append(i)
            j, length = glove_embed(i)
            if length<= 0:
                embeddings.append(None)
                continue
            if email%1000==0 and printc:
                print("Iteration: ", email)
                printc=False
            try:
                embeddings.append(j/length)
            except:
                print(i, email)
    assert len(embeddings) == len(sentence_corpus)
    return embeddings, sentence_corpus

def glove_search(query, embeddings, sentence_corpus, indx_to_email, top_n=2, is_preprocess=False):
    """
    Input: query to search, Sentence corpus to search in , top n results\n
    Searches with cosing similarity among the embeddings\n
    Output: most similar items to query\n
    """
    q_emb = []
    """Store the Embddings of the query"""
    print("Query:", "\33[31m" + query + "\033[0m")
    if is_preprocess:
        query=preprocess(query)
        print("Preprocessed Query:", "\33[31m" + query + "\033[0m ")
    j, length = glove_embed(query.lower())
    
    if length<= 0:
        print("Sent not found",  i)
        q_emb.append(None)
    try:
        q_emb.append(j/length)
    except:
        print(i, email)
    similarity = []
    for emb in range(len(embeddings)):
        score = cosine_score(embeddings[emb], q_emb[0])
        similarity.append(cosine_score(embeddings[emb], q_emb[0]))
    sim = np.argsort(similarity)[::-1][:top_n]
    for i in sim:
            print("\33[33m" + "Cosine Similarity: {:.3f} \t Email {} \t Line {} \033[0m".format(similarity[i], indx_to_email[i][1], i-indx_to_email[i][0]))
            print("\33[34m" + str(sentence_corpus[indx_to_email[i][1]][i-indx_to_email[i][0]]) + '\033[0m',  "\n")
