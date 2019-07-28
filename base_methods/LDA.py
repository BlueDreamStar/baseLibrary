from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
import os

def write_lines(string, saveFile):
    open("%s" %saveFile, "a").write(string+"\n")

### 
def save_topics(model, feature_names, saveFile, topic_nums, top_words_nums):
    # saveFile= "%s/LDA_TopWords_Topic%s.txt" %(saveFileDir,topic_nums)
    ## Save Topics Top Words
    for topic_idx, topic in enumerate(model.components_):
        # print(saveFile)
        write_lines("Topic %s:" % (topic_idx), saveFile)
        for i in topic.argsort()[:-top_words_nums - 1:-1]:
            # print(feature_names[i])
            write_lines(feature_names[i].replace("\n",""), saveFile)
        write_lines("",saveFile)
        
def get_labels(poiTagDir):
    f = open(poiTagDir,"r")
    fLines = f.readlines()
    poiTag=[]
    for l in fLines:
        poiTag.append(l)
    return poiTag
        
def run_lda(documents, feature_names, saveFileDir, topic_nums = 10, top_words_nums = 20):
    lda = LatentDirichletAllocation(n_topics=topic_nums, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(documents)
    
    saveFileHeader = "%s/LDA_TopWords_Topic%s" %(saveFileDir,topic_nums)
    ### save lda outcomes
    saveFile= "%s.txt" %(saveFileHeader)
    if os.path.exists(saveFile):
        os.remove(saveFile)
    
    ## Save Topic top words
    save_topics(lda, feature_names, saveFile, topic_nums, top_words_nums)    
    
    ## Save Topic-words Matrix
    np.savetxt("%s_Topic_Words_matrix.txt" %(saveFileHeader), lda.components_, fmt="%.6f")
    
    ## Save documents-topics
    documents_topics = lda.transform(documents)
    np.savetxt("%s_Document_Topics_matrix.txt" %(saveFileHeader), documents_topics, fmt="%.6f")
    np.savetxt("%s_Document_Topic.txt" %(saveFileHeader), np.argmax(documents_topics,axis=1).reshape(len(documents_topics),1), fmt="%d")
    
    ## Save perplexity
    # print(lda.perplexity(documents))
    np.savetxt("%s_perplexity.txt" %(saveFileHeader), [-1, lda.perplexity(documents)], fmt="%.6f")
    
