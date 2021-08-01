# coding=utf-8
import random
import os
if __name__ == "__main__":
    DATA_PATH='argoLog_pro_finish.txt'
    DATA_PATH_SF='argoLog_pro_finish_sf.txt'
    sentences=[]
    for sentence in open(DATA_PATH):
        sentences.append(sentence)
    sentences=[s.encode('utf-8').split() for s in sentences]
    # print sentences[0]
    #打乱顺序
    random.shuffle(sentences)

    # with open(DATA_PATH_SF,'w') as f:
    #     for sentence in sentences:
    #         print sentence
    #         f.write(sentence)
    #         f.write("\n")
    #     f.close()
    with open(DATA_PATH_SF,'w') as f:
        content=''
        for sentence in sentences:
            for word in sentence:
                content+=word
                content+=' '
            content+='\n'
        f.write(content)
        f.close()


