#https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing 
#(Pretrained word2vec model link)
import gensim
from gensim.models import word2vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
v_apple = word_vectors["apple"] 
v_mango = word_vectors["india"]

print(v_apple.shape)
print(v_mango.shape)

cosine_similarity([v_mango],[v_apple])

import numpy as np
#1. Find the Odd One Out

def odd_one_out(words):
    """Accepts a list of words and returns the odd word"""
    
    # Generate all word embeddings for the given list
    all_word_vectors = [word_vectors[w] for w in words]
    avg_vector = np.mean(all_word_vectors,axis=0)
    print(avg_vector.shape)
    
    #Iterate over every word and find similarity
    odd_one_out = None
    min_similarity = 1.0 #Very high value
    
    for w in words:
        sim = cosine_similarity([word_vectors[w]],[avg_vector])
        if sim < min_similarity:
            min_similarity = sim
            odd_one_out = w
    
        print("Similairy btw %s and avg vector is %.2f"%(w,sim))
            
    return odd_one_out

input_1 = ["apple","mango","juice","party","orange"] 
input_2 = ["music","dance","sleep","dancer","food"]        
input_3  = ["match","player","football","cricket","dancer"]
input_4 = ["india","paris","russia","france","germany"]

odd_one_out(input_1)
odd_one_out(input_2)
odd_one_out(input_3)
odd_one_out(input_4)

#2. Word Analogies Task
#In the word analogy task, we complete the sentence "a is to b as c is to __". An example is 'man is to woman as king is to queen' . In detail, we are trying to find a word d, such that the associated word vectors ea,eb,ec,ed are related in the following manner: eb−ea≈ed−ec. We will measure the similarity between eb−ea and ed−ec using cosine similarity.
#Word2Vec
"""
man -> woman ::    prince -> princess
italy -> italian ::    spain -> spanish
india -> delhi ::  japan -> tokyo
man -> woman ::    boy -> girl
small -> smaller ::    large -> larger

Try it out
man -> coder :: woman -> ______?
"""

type(word_vectors.vocab)

word_vectors["man"].shape

def predict_word(a,b,c,word_vectors):
    """Accepts a triad of words, a,b,c and returns d such that a is to b : c is to d"""
    a,b,c = a.lower(),b.lower(),c.lower()
    
    # similarity |b-a| = |d-c| should be max
    max_similarity = -100 
    
    d = None
    
    words = word_vectors.vocab.keys()
    
    wa,wb,wc = word_vectors[a],word_vectors[b],word_vectors[c]
    
    #to find d s.t similarity(|b-a|,|d-c|) should be max
    
    for w in words:
        if w in [a,b,c]:
            continue
        
        wv = word_vectors[w]
        sim = cosine_similarity([wb-wa],[wv-wc])
        
        if sim > max_similarity:
            max_similarity = sim
            d = w
            
    return d

triad_2 = ("man","woman","prince")
predict_word(*triad_2,word_vectors)

#Method 2
#Using the Most Similar Method

word_vectors.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)