#!/usr/bin/env python

import random
import nltk
import sklearn
import os

from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#nltk.download('punkt')

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

def getData(tempData, tempDataLabel):
    with open("./dados/avaliacoes_positivas.txt") as f:
        for i in f: 
            tempData.append(i) 
            tempDataLabel.append('positivo')

    with open("./dados/avaliacoes_negativas.txt") as f:
        for i in f: 
            tempData.append(i)
            tempDataLabel.append('negativo') 

def setData(avaliacao, sentimento):
    if(sentimento == 'positiva'):
        with open("./dados/avaliacoes_positivas.txt", "a+") as file:
            file.seek(0)
            content = file.read(100)
            if len(content) > 0:
                file.write("\n")
            file.write("\"" + avaliacao + "\"")
    else:
        with open("./dados/avaliacoes_negativas.txt", "a+") as file:
            file.seek(0)
            content = file.read(100)
            if len(content) > 0:
                file.write("\n")
            file.write("\"" + avaliacao + "\"")

def getVetorizer():
    return CountVectorizer(
        analyzer = 'word',
        lowercase = False,
    )

def getFeaturesArray(vectorizer, data):
    features = vectorizer.fit_transform(
        data
    ).toarray()
    return features

def getFit():

    data = []
    
    data_labels = []    

    getData(data,data_labels)

    vectorizer = getVetorizer()

    features = getFeaturesArray(vectorizer,data)

    X_train, X_test, y_train, y_test  = train_test_split(
            features, 
            data_labels,
            train_size=0.75, 
            random_state=0)


    log_model = LogisticRegression()

    log_model = log_model.fit(X=X_train, y=y_train)

    y_pred = log_model.predict(X_test)

    j = random.randint(0,len(X_test)-7)

    for i in range(j,j+7):
        #print(y_pred[0])
        ind = features.tolist().index(X_test[i].tolist())
        #print(data[ind].strip())

    #print(accuracy_score(y_test, y_pred))  

    pos = []

    with open("./dados/avaliacoes_positivas.txt") as f:
        for i in f: 
            pos.append([format_sentence(i), 'positivo'])

    neg = []
    with open("./dados/avaliacoes_negativas.txt") as f:
        for i in f: 
            neg.append([format_sentence(i), 'negativo'])

    training = pos[:int((.75)*len(pos))] + neg[:int((.75)*len(neg))]
    
    test = pos[int((.75)*len(pos)):] + neg[int((.75)*len(neg)):]
    
    classifier = NaiveBayesClassifier.train(training)
    
    print("Score de Precisão: " + str(accuracy(classifier, test)))

    return classifier

def getTest(classifier):
    with open("./dados/avaliacoes_teste.txt") as f:
        for i in f:
            print(i, end=" ")
            print("(" + classifier.classify(format_sentence(i)) + ")")
            print("\n")

def setTest(avaliacao):
    with open("./dados/avaliacoes_teste.txt", "a+") as file:
        file.seek(0)
        content = file.read(100)
        if len(content) > 0:
            file.write("\n")
        file.write(avaliacao)

def main():     
    os.system('clear')
    print()
    print("**********************/ Classificador de Sentimento - NaiveBayesClassifier /**********************")
    print()

    opt = "-1"
    classifier = getFit()
    
    while opt != "7":
        print("\n1 - Classifiar nova avaliação\n2 - Inserir dado para treino(positivo) \n3 - Inserir dado para treino(Negativo)\n4 - Treinar\n5 - Inserir dados para testes\n6 - Rodar arquivo teste\n7 - Sair")
        print()
        opt = input("Operação: ")
        print()
        if opt == "1":
            os.system('clear')
            avaliacao = input("Escreva uma avaliação para ser classificada: ")
            print(avaliacao, end=" - ")
            print(classifier.classify(format_sentence(avaliacao)))
        elif opt == "2":
            os.system('clear')
            avaliacao = input("Escreva uma avaliação Positiva para ser incluida no arquivo de treino: ")
            setData(avaliacao,"positiva")
        elif opt == "3":
            os.system('clear')
            avaliacao = input("Escreva uma avaliação Negativa para ser incluida no arquivo de treino: ")
            setData(avaliacao,"negativa")        
        elif opt == "4":
            os.system('clear')
            classifier = getFit()
            print("Treino Concluido")
            print(classifier.show_most_informative_features())
        elif opt == "5":
            os.system('clear')
            avaliacao = input("Escreva uma avaliação para ser incluida no arquivo de teste: ")
            setTest(avaliacao)       
        elif opt == "6":
            os.system('clear')
            getTest(classifier)
    
    os.system('clear')         

if __name__ == '__main__':
    main()