from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from IPython.display import Markdown, display, clear_output
import numpy as np
import _pickle as cPickle
from pathlib import Path
import spacy
from spacy import displacy
from App.models import Contacts
from django.contrib import messages
from datetime import datetime
import os
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors




model = KeyedVectors.load("App/word2vec.model")
Question = []
Answer = []



# Create your views here.
def Home(request):

        
        def dumpPickle(fileName, content):
            pickleFile = open(fileName, 'wb')
            cPickle.dump(content, pickleFile, -1)
            pickleFile.close()

        def loadPickle(fileName):    
            file = open(fileName, 'rb')
            content = cPickle.load(file)
            file.close()
            
            return content
        
        def pickleExists(fileName):
            file = Path(fileName)
            
            if file.is_file():
                return True
            
            return False


        # ## *Extract all words from plain text and generate it's features*

        # In[6]
        nlp = spacy.load('en_core_web_sm')

        #Extract answers and the sentence they are in
        def extractAnswers(qas, doc):
           

            answers = []

            senStart = 0
            senId = 0

            for sentence in doc.sents:
                senLen = len(sentence.text)

                for answer in qas:
                    answerStart = answer['answers'][0]['answer_start']

                    if (answerStart >= senStart and answerStart < (senStart + senLen)):
                        answers.append({'sentenceId': senId, 'text': answer['answers'][0]['text']})

                senStart += senLen
                senId += 1
            
            return answers

        #TODO - Clean answers from stopwords?
        def tokenIsAnswer(token, sentenceId, answers):
            for i in range(len(answers)):
                if (answers[i]['sentenceId'] == sentenceId):
                    if (answers[i]['text'] == token):
                        return True
            return False

        #Save named entities start points

        def getNEStartIndexs(doc):
            neStarts = {}
            for ne in doc.ents:
                neStarts[ne.start] = ne
                
            return neStarts 

        def getSentenceStartIndexes(doc):
            senStarts = []
            
            for sentence in doc.sents:
                senStarts.append(sentence[0].i)
            
            return senStarts
            
        def getSentenceForWordPosition(wordPos, senStarts):
            for i in range(1, len(senStarts)):
                if (wordPos < senStarts[i]):
                    return i - 1
                
        def addWordsForParagrapgh(newWords, text):
            doc = nlp(text)

            neStarts = getNEStartIndexs(doc)
            senStarts = getSentenceStartIndexes(doc)
            
            #index of word in spacy doc text
            i = 0
            
            while (i < len(doc)):
                #If the token is a start of a Named Entity, add it and push to index to end of the NE
                if (i in neStarts):
                    word = neStarts[i]
                    #add word
                    currentSentence = getSentenceForWordPosition(word.start, senStarts)
                    wordLen = word.end - word.start
                    shape = ''
                    for wordIndex in range(word.start, word.end):
                        shape += (' ' + doc[wordIndex].shape_)

                    newWords.append([word.text,
                                    0,
                                    0,
                                    currentSentence,
                                    wordLen,
                                    word.label_,
                                    None,
                                    None,
                                    None,
                                    shape])
                    i = neStarts[i].end - 1
                #If not a NE, add the word if it's not a stopword or a non-alpha (not regular letters)
                else:
                    if (doc[i].is_stop == False and doc[i].is_alpha == True):
                        word = doc[i]

                        currentSentence = getSentenceForWordPosition(i, senStarts)
                        wordLen = 1

                        newWords.append([word.text,
                                        0,
                                        0,
                                        currentSentence,
                                        wordLen,
                                        None,
                                        word.pos_,
                                        word.tag_,
                                        word.dep_,
                                        word.shape_])
                i += 1

        def oneHotEncodeColumns(df):
            columnsToEncode = ['NER', 'POS', "TAG", 'DEP']

            for column in columnsToEncode:
                one_hot = pd.get_dummies(df[column])
                one_hot = one_hot.add_prefix(column + '_')

                df = df.drop(column, axis = 1)
                df = df.join(one_hot)
            
            return df


        # ## *Predict whether a word is a keyword* 

        # In[7]:


        def generateDf(text):
            words = []
            addWordsForParagrapgh(words, text)

            wordColums = ['text', 'titleId', 'paragrapghId', 'sentenceId','wordCount', 'NER', 'POS', 'TAG', 'DEP','shape']
            df = pd.DataFrame(words, columns=wordColums)
            
            return df


        # In[8]:


        def prepareDf(df):
            #One-hot encoding
            wordsDf = oneHotEncodeColumns(df)


            #Add missing colums 
            predictorFeaturesName = 'Data/nb-predictor-features.pkl'
            featureNames = loadPickle(predictorFeaturesName)

            for feature in featureNames:
                if feature not in wordsDf.columns:
                    wordsDf[feature] = 0    
                        
            #Drop unused columns
            columnsToDrop = ['text', 'titleId', 'paragrapghId', 'sentenceId', 'shape', 'isAnswer']
            wordsDf = wordsDf.drop(columnsToDrop, axis = 1)


            return wordsDf


        # In[9]:


        def predictWords(wordsDf, df):
            
            predictorPickleName = 'Data/nb-predictor.pkl'
            predictor = loadPickle(predictorPickleName)
            
            y_pred = predictor.predict_proba(wordsDf)

            labeledAnswers = []
            for i in range(len(y_pred)):
                labeledAnswers.append({'word': df.iloc[i]['text'], 'prob': y_pred[i][0]})
            
            return labeledAnswers


        # ## *Extract questions*

        # In[10]:


        def blankAnswer(firstTokenIndex, lastTokenIndex, sentStart, sentEnd, doc):
            leftPartStart = doc[sentStart].idx
            leftPartEnd = doc[firstTokenIndex].idx
            rightPartStart = doc[lastTokenIndex].idx + len(doc[lastTokenIndex])
            rightPartEnd = doc[sentEnd - 1].idx + len(doc[sentEnd - 1])
            
            question = doc.text[leftPartStart:leftPartEnd] + '_____' + doc.text[rightPartStart:rightPartEnd]
            
            return question


        # In[11]:


        def addQuestions(answers, text):
            Question.clear()
            Answer.clear()
            doc = nlp(text)
            currAnswerIndex = 0
            qaPair = []

            #Check wheter each token is the next answer
            for sent in doc.sents:
                for token in sent:
                    
                    #If all the answers have been found, stop looking
                    if currAnswerIndex >= len(answers):
                        break
                    
                    #In the case where the answer is consisted of more than one token, check the following tokens as well.
                    answerDoc = nlp(answers[currAnswerIndex]['word'])
                    answerIsFound = True
                    
                    for j in range(len(answerDoc)):
                        if token.i + j >= len(doc) or doc[token.i + j].text != answerDoc[j].text:
                            answerIsFound = False
                
                    #If the current token is corresponding with the answer, add it 
                    if answerIsFound:
                        question = blankAnswer(token.i, token.i + len(answerDoc) - 1, sent.start, sent.end, doc)
                        
                        qaPair.append({'question' : question, 'answer': answers[currAnswerIndex]['word'], 'prob': answers[currAnswerIndex]['prob']})
                        
                        currAnswerIndex += 1
                        
            return qaPair


        # In[12]:


        def sortAnswers(qaPairs):
            orderedQaPairs = sorted(qaPairs, key=lambda qaPair: qaPair['prob'])
            
            return orderedQaPairs    


        # ## *Distractors*
        
        def addDistractors(qaPairs, count):
            if not model:
                print("Glove embeddings not found. Please download and place them in the following path: ")
            
            for qaPair in qaPairs:
                distractors = generate_distractors(qaPair['answer'], count)
                qaPair['distractors'] = distractors
            
            return qaPairs
        def generate_distractors(answer, count):
            answer = str.lower(answer)
            closestWords = model.most_similar(positive=[answer], topn=count)
            
            ##Extracting closest words for the answer. 
            try:
                closestWords = model.most_similar(positive=[answer], topn=count)
                print(closestWords)
            except:
                #In case the word is not in the vocabulary, or other problem not loading embeddings
                return print(closestWords)

            #Return count many distractors
            distractors = list(map(lambda x: x[0], closestWords))[0:count]
            
            return distractors




        # # Main function

        # In[16]:


        def generateQuestions(text, count):
            
            # Extract words 
            df = generateDf(text)
            wordsDf = prepareDf(df)
           
            # Predict 
            labeledAnswers = predictWords(wordsDf, df)
            
            # Transform questions
            qaPairs = addQuestions(labeledAnswers, text)
            
            # Pick the best questions
            orderedQaPairs = sortAnswers(qaPairs)
            
            # Generate distractors
            questions = addDistractors(orderedQaPairs[:count], 4)
            # Print
            for i in range(count):
                Question.append(questions[i]['question'])
               
                Answer.append(questions[i]['answer'])
                Question.append(questions[i]['answer'])
                
                
                for distractor in questions[i]['distractors']: 
                    Question.append(distractor)
                   
                                    
        if request.method == 'POST':
            text =request.POST.get('text')
            number = int(request.POST.get('number'))
            generateQuestions(text,number)
            context = {
                'Questions': Question,  
                'Answers':Answer,
            }
            return render(request,'QuestionsGenrated.HTML',context)
        return render(request,'Processing.HTML')

def Contact(request):
    if (request.method == 'POST'):
        name = request.POST.get('name')
        email=request.POST.get('email')
        subject = request.POST.get('desc')
        contacts = Contacts(Name=name,Email=email,Des=subject,Date = datetime.now())
        print(request.POST.items())
        contacts.save()
        messages.success(request,"Your message was sent ")
    return render(request,'ContactUS.HTML')
  

               
               
               
            
              
              
        
def Donations(request):
    return render(request,'Donations.HTML')
def Mainpage(request):
    return render(request,'Info.HTML')
# %%
