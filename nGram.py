"""
  Consumes an input corpus and a test set, builds n-gram models,
  then tests those models against the test data. Test/training split is
  the usual 10/90.

  I wrote this independent of the other projects because its so much 
  easier in python, for generating n-gram baselines for comparison with 
  other methods.


  Results (somewhat reliable, 12/22):
    Scores (21400 predictions)
      Bool:   15.6308411215%
      Real:   23.1992304123%
      Top10:  39.3971962617%
      PP:     99.7852299488%
      Recall: 66.9065420561%
      AvgLen: 4.52722097023 (avg correct word length for ksr measure, of top10 words only)
    vectormodel >>
    Scores (21400 predictions)
      Bool:   15.1588785047%
      Real:   23.1098681484%
      Top10:  39.5420560748%
      PP:     9.8891144%
      Recall: 66.9065420561%
      AvgLen: 4.58484991728 (avg correct word length for ksr measure, of top10 words only)
    ensemble    >>
    Scores (21400 predictions)
      Bool:   15.6355140187%
      Real:   23.2385181941%
      Top10:  39.1448598131%
      Recall: 66.9065420561%
      AvgLen: 4.55162946162 (avg correct word length for ksr measure, of top10 words only)



  If valid, these results are also slightly pessimistic, due to noise in the training data.
  Additionally, the enron dataset is inherently difficult due to the nature of the language
  in a business and engineering context; qualitatively, the branching factor is much higher,
  likely because such language is more descriptive/external than internal/personal. Make an
  analogy of this in the paper.

  These results (if accurate) are surprising, and promising. They are much better
  than general language model performance, implying that personal language models
  are much easier to estimate than language models. This is intuitive, since most
  people reuse the same vocabulary, phrase structures, and so on, and also speak
  in recurring contexts.

  Based on this, a todo: evaluate entropy of personal language. (Cross-entropy as well?)
  Since personal language is easier to predict, it is possible there are other unique attributes
  involved, besides just a more limited vocabulary. For instance, recurrent contexts (people, places,
  subjects, etc) give rise to the same recurrent phrase structures, lexicons, and so on.

  Another todo: confirm these results for another individual. Confirm them with c++.

  


"""

import math

#score results. Remember results are in -log2-space, hence the conversions
def scoreNgram(actual,results,scores):
  j = 1
  pActual = 0.0
  found = False

  normal = 0.0
  for res in results:
    normal += res[0]

  while j <= len(results) and found == False:
    resultTup = results[j-1]
  #for resultTup in results:
    #scores are by absolute match, or by substring matches if over 5 letters; note the danger of substring matches: "thorough" matches "thoroughly"
    if actual == resultTup[1]: # or (len(actual) > 5 and actual == resultTup[1][0:len(actual)]):
      found = True
      #print "hit: "+actual+"|"+resultTup[1]
      pActual = resultTup[0]
      scores["recall"] += 1.0
      scores["pp"] += ((pActual - results[0][0]) / normal)
      scores["real"] += (1.0 / float(j))
      if j == 1:
        scores["bool"] += 1.0
        #scores["real"] += 1.0
      if j <= 10:
        scores["avgwordlen"] += float(len(actual))
        scores["top10"] += 1.0
    j+=1

  return scores

#score the vector results, where each result is a list as ["w1 w2 w3",0,0.0] as [postStr,postStr_count,cosine_dist]
def scoreVector(actual,results,vecScores):
  j = 1
  found = False

  normal = 0.0
  for res in results:
    normal += res[2]

  while (j <= len(results)) and (found == False):
    resultTup = results[j-1]
    #for resultTup in results:
    #scores are by absolute match, or by substring matches if over 5 letters; note the danger of substring matches: "thorough" matches "thoroughly"
    if actual == resultTup[0].split(" ")[0]: # or (len(actual) > 5 and actual == resultTup[1][0:len(actual)]):
      found = True
      #print "hit: "+actual+"|"+resultTup[1]
      vecScores["recall"] += 1.0
      vecScores["pp"] += ((results[0][2] - resultTup[2]) / normal)
      vecScores["real"] += (1.0 / float(j))
      if j == 1:
        vecScores["bool"] += 1.0
        #vecScores["real"] += 1.0
      if j <= 10:
        vecScores["avgwordlen"] += float(len(actual))
        vecScores["top10"] += 1.0
    j+=1

  return vecScores




# Slightly different data model than score(), since ensemble results only hold their merged rank as their "score"
# The intent of the ensemble is only to see if raw accuracy improves by merging.
def scoreEnsemble(actual,results,scores):
  j = 1
  found = False

  while (j <= len(results)) and (found == False):
    resultTup = results[j-1]
  #for resultTup in results:
    #scores are by absolute match
    if actual == resultTup[0]: # or (len(actual) > 5 and actual == resultTup[1][0:len(actual)]):
      found = True
      #print "hit: "+actual+"|"+resultTup[1]
      scores["recall"] += 1.0
      scores["real"] += (1.0 / float(j))
      if j == 1:
        scores["bool"] += 1.0
        #scores["real"] += 1.0
      if j <= 10:
        scores["avgwordlen"] += float(len(actual))
        scores["top10"] += 1.0
    j+=1

  return scores

#vector and ensemble models
def printScores_2(scores,i):
    print "Scores ("+str(i)+" predictions)"
    print "  Bool:   "+str((scores["bool"] / float(i)) * 100)+"%"
    print "  Real:   "+str((scores["real"] / float(i)) * 100)+"%"
    print "  Top10:  "+str((scores["top10"] / float(i)) * 100)+"%"
    if scores["pp"] > 0:
      print "  PP:     "+str((scores["pp"] / scores["recall"]) * 100)+"%"
    print "  Recall: "+str((scores["recall"] / float(i)) * 100)+"%"
    if float(scores["top10"]) > 0:
      print "  AvgLen: "+str(scores["avgwordlen"] / float(scores["top10"]))+" (avg correct word length for ksr measure, of top10 words only)"

#ngram model only
def printScores_1(scores,i):
  if i > 0:
    print "Scores ("+str(i)+" predictions)"
    print "  Bool:   "+str((scores["bool"] / float(i)) * 100)+"%"
    print "  Real:   "+str((scores["real"] / float(i)) * 100)+"%"
    print "  Top10:  "+str((scores["top10"] / float(i)) * 100)+"%"
    print "  PP:     "+str((math.pow(2.0, -1.0 * (scores["pp"]) / float(i))) * 100)+"%"
    print "  Recall: "+str((scores["recall"] / float(i)) * 100)+"%"
    if float(scores["top10"]) > 0:
      print "  AvgLen: "+str(scores["avgwordlen"] / float(scores["top10"]))+" (avg correct word length for ksr measure, of top10 words only)"

# get sum distance of this three word vector to the current context, based on weighted-cosine model data
def getSumDistance(postStr,sequence,i,coTable,table):
  j = 0
  tokens = postStr.split(" ")
  dist = 0.0
  for tok in tokens:
    j = 0
    while j < 15:
      dist += getSimilarity(tok,sequence[i-j],coTable,table)
      j+=1

  return dist

#sort result tuples by cosine distance
def byCosineDist(res1,res2):
  if res1[2] < res2[2]:
    return 1
  if res1[2] == res2[2]:
    return 0
  return -1

#results is a list of lists, where sublists are [prediction,prob,rank] (where rank will be used for ensemble merging)
def predictVectorModel(vecModel,coTable,testData,i,table):
  results = []
  result = []
  #get all the examples
  if vecModel.get(testData[i],-1) != -1:
    for example in vecModel[testData[i]]:
      result = example
      result[2] = getSumDistance(result[0],testData,i,coTable,table) * float(result[1])
      results.append(result)
    #print "vec presorted:",results
    results.sort(byCosineDist)
    #print "vec sorted:",results
  return results

#predict via linear interpolation
def predictNgramModel(oneModel,twoModel,threeModel,fourModel,wordSeq,i,lambdas):

  zeroProb = 4.0
  results = []
  #build the model queries
  oneQ = wordSeq[i]
  twoQ = wordSeq[i]
  threeQ = wordSeq[i-1] + " " + wordSeq[i]
  fourQ = wordSeq[i-2] + " " + wordSeq[i-1] + " " + wordSeq[i]

  #declare the sub dictionaries (each model for n > 1 is of type:  key<string> -> value<dict<string:float>>)
  twoDict = {}
  threeDict = {}
  fourDict = {}
  #get all the subdicts, for n > 1
  #get two-gram subdict
  if twoModel.get(twoQ,-1) != -1:
    twoDict = twoModel[twoQ]
  #get three-gram dict
  if threeModel.get(threeQ,-1) != -1:
    threeDict = threeModel[threeQ]
  #get four-gram dict
  if fourModel.get(fourQ,-1) != -1:
    fourDict = fourModel[fourQ]
  # all dicts defined

  #declare the log-probability vals
  Pone = 0.0 
  Ptwo = 0.0
  Pthree = 0.0
  Pfour = 0.0  
  #lookup all queries in all models, and merge subsets (there's signficant overlap here; this method is meant to merge all
  # (in reality, if all models were trained off the same data, if one-gram lookup fails, they all fail)
  if oneModel.get(oneQ,-1) != -1:
    Pone = oneModel[oneQ]

  if Pone == 0:
    return results

  # Note at this point, fourDict, threeDict, etc, are first order word dictionaries: word -> conditional probability
  #recursively sum (there's no explicit recursion here; just recognize the recursive structure of the n-gram models: 4-3-2-1, 3-2-1, 2-1, 1)  
  for key4 in fourDict.keys():
    sumVal = fourDict[key4] * lambdas[3]
    sumVal += oneModel.get(key4,zeroProb) * lambdas[0]
    for key3 in threeDict.keys():
      if key3 == key4:
        sumVal += threeDict[key3] * lambdas[2]
    for key2 in twoDict.keys():
      if key2 == key4:
        sumVal += twoDict[key2] * lambdas[1]
    results.append((sumVal,key4))    
  #sum for three-gram and lesser models
  for key3 in threeDict.keys():
    sumVal = threeDict[key3] * lambdas[2]
    sumVal += oneModel.get(key3,zeroProb) * lambdas[0]
    for key2 in twoDict.keys():
      if key2 == key3:
        sumVal += twoDict[key2] * lambdas[1]
    results.append((sumVal,key3))    
  #sum for two-gram and lesser model
  for key2 in twoDict.keys():
    sumVal = twoDict[key2] * lambdas[1]
    sumVal += oneModel.get(key2,zeroProb) * lambdas[0]
    results.append((sumVal,key2))    

  return results  #tuple in result list vary in size, but all hold sumVal in [0] (as a sum log-space val)


#sort tuple list by frequency (second item in two-tuple)
def byFreq(x,y):
  if x[1] >= y[1]:
    return -1
  return 1

#sort negated log-probabilities, ascending (lower log-probs better)
def compareLogs(x,y):
  if x[0] <= y[0]:
    return -1
  return 1

def allNum(tok):
  for char in tok:
    if char not in "$%+-0987654321/\\":
      return False
  return True

def filterTokens(toks):
  tokens = []
  for tok in toks:
    newTok = tok.replace("!","").replace(":","").replace(";","").replace(",","").replace(".","").replace("?","").replace("'","").replace("\"","").replace("(","").replace(")","")
    if len(newTok) > 0 and not allNum(newTok):
      tokens.append(newTok.lower())
  #print tokens
  return tokens


def getWordSequence(ifile):
  lines = ifile.readlines()
  tokens = []
  for line in lines:
    if "SUBJECT:" not in line:
      if "TO:" not in line:
        if "DATE:" not in line:
          if "BODY:" not in line:
            if "EOF" not in line:
              tokens += line.strip().replace("-"," ").split(" ")
  tokens = filterTokens(tokens)
  print "n-training tokens: ",str(len(tokens))
  return tokens

#driver for dictToLog2Space
def subdictToLog2Space(dictionary):
  normal = 0.0
  for key in dictionary.keys():
    normal += float(dictionary[key])

  for key in dictionary.keys():
    freq = float(dictionary[key])
    logval = math.log(freq/normal,2.0)
    if logval != 0.0:
      dictionary[key] = logval * -1.0
    else:
      dictionary[key] = logval
  return dictionary

#converts all conditional freq in an outer dict to log2 space
def dictToLog2Space(dictionary):
  for key in dictionary.keys():
    dictionary[key] = subdictToLog2Space(dictionary[key])
  return dictionary

#to save space, only store integers instead of whole strings. 
#assume "table" is an object of type [{the id->word table},{the word->id table},counter], where counter is the current id counter val
def wordToId(word,table):
  if table[1].get(word,-1) == -1: #alloc a new id if none found, and increment the id counter
    table[1][word] = table[2]
    table[2] += 1
  return table[1][word]
#counterpart to last
def idToWord(iD,table):
  if table[0].get(iD,-1) == -1:
    print "ERROR id "+str(iD)+" not found in table!"
    return 0
  else:
    return table[0][iD]

#fetch similarity of word 1 and word 2 from coDistance matrix
def getSimilarity(w1,w2,coTable,table):
  w1_id = wordToId(w1,table)
  w2_id = wordToId(w2,table)

  #check for keys in each order: [w1][w2] and [w2][w1]
  if coTable.get(w1_id,-1) != -1:
    if coTable[w1_id].get(w2_id,-1) != -1:
      return coTable[w1_id][w2_id]
  if coTable.get(w2_id,-1) != -1:
    if coTable[w2_id].get(w1_id,-1) != -1:
      return coTable[w2_id][w1_id]
  return 0.0

#sort ascending (lower ranks better)
def compEnsembleItem(item1, item2):
  if item1[1] > item2[1]:
    return 1
  if item1[1] == item2[1]:
    return 0
  return -1

# Merge results of n-gram and vector predictions to form an ensemble, where the rank of
# each item (result) is: (rank(ngram) + rank(vector)) / 2. If one model does not contain
# an item that the other does, then the ensemble rank is 2 * rank(known model) / 2 == rank(known model)
#   precondition: both results are sorted descending, most likely prediction at [0]
def predictEnsemble(ngResults,vecResults):
  results = []
  i = 0
  while i < len(ngResults):
    res = [ngResults[i][1],i]
    j = 0
    match = False
    while j < len(vecResults) and match == False:
      if vecResults[j][0].split(" ")[0] == ngResults[i][1]:
        res[1] += j
        match = True
      j+=1
    if res[1] == i:
      res[1] *= 2
    else:
      res[1] /= 2
    i+=1
    results.append(res)

  results.sort(compEnsembleItem)
  #print "sorted ensemble:",results
  #raw_input()

  return results

#given a dictionary of (word,frequency) key/val pairs, prints the items in sorted order
def printMostAccurateWords(freqDict):
  tuplist = []
  for word in freqDict.keys():
    tuplist.append((word,freqDict[word]))
  tuplist.sort(byFreq)

  print freqDict
  print "the most accurate words:"
  i = 0
  while i < 50 and i < len(tuplist):
    print tuplist[i][0]+":"+str(tuplist[i][1])
    i += 1



#build codist table from pre-made file. table is some object (a manager really) containing mapping back and forth from word-id and id-word
def buildCoDistTable(coFile,coTable,table):
  i = 0
  lines = coFile.readlines()
  print "processing "+str(len(lines))+" lines of codist file..."
  for line in lines:
    tokens = line.split("\t")
    w1_id = wordToId(tokens[0],table)
    w2_id = wordToId(tokens[1],table)
    similarity = float(tokens[2])
    if coTable.get(w1_id,-1) == -1:
      coTable[w1_id] = {w2_id:similarity}
      i+=1
    else:
      coTable[w1_id][w2_id] = similarity
  print "codist file processing complete, nkeys="+str(len(coTable.keys()))+"  verify equals count: "+str(i)

#build vector model, where key=pre, val=(postStr,count-postStr)
def buildVectorModel(vectorModel,trainingData):
  i = 0
  length = len(trainingData)
  while i < (length - 3):
    pre = trainingData[i]
    post = [trainingData[i+1]+" "+trainingData[i+2]+" "+trainingData[i+3],1,0.0]
    if vectorModel.get(pre,-1) == -1:
      vectorModel[pre] = [post]
    else:
      found = False
      for valPair in vectorModel[pre]:
        #print "vp=",valPair," post=",post
        if valPair[0].split(" ")[0] == post[0].split(" ")[0]: #increment bigram counts
          valPair[1] += 1
          found = True
      if found == False:
        vectorModel[pre] += [post]
    i+=1
    if i % 5000 == 4999:
      print "\r"+str(100 * (float(i) / float(length)))+"% completed     "


#####################################  main  #####################################

oneGramModel = {}
twoGramModel = {}
threeGramModel = {}
fourGramModel = {}

#set linear interpolation lambdas


"""
  this file contains about 180k words from an enron user, which
  in reality is too little data for an n-gram model. This is being
  used only for baselining a comparison of personal-ngram models with
  other methods. The assumption is that a smaller dataset may be sufficient
  for estimating a personal language model instead of a general language
  model (which requires millions of words). Thus, use one of the most prolific
  emailers in the enron dataset, to compare personal-model methods.
"""
trainFile = open("/home/jesse/Desktop/TeamGleason/src/PersonalModel/germanyRawWords.txt", "r")
testFile = open("/home/jesse/Desktop/TeamGleason/src/PersonalModel/germanyRawTest.txt", "r")
coDistFile = open("/home/jesse/Desktop/TeamGleason/src/PersonalModel/personal_coocWeightedCosine.txt", "r")

trainingData = getWordSequence(trainFile)
testData= getWordSequence(testFile)

trainFile.close()
testFile.close()

print "building models..."
#now build the models from filtered sequence. tokens are lowercased, 'flattened' as much as possible

idToWordTable = {}
wordToIdTable = {}
wordIdManager = [idToWordTable,wordToIdTable,1]
coDistTable = {}
buildCoDistTable(coDistFile,coDistTable,wordIdManager)
coDistFile.close()

#build vector model: key=preString, val=(w1,w2,w3,count-pre) (w1,w2,w3 is really the postStr
print "building vector model..."
found = False
vectorModel = {}
buildVectorModel(vectorModel,trainingData)
print "vector model complete... vectorModel.size="+str(len(vectorModel.keys()))
#example model format of key/vals:
#key=second val=[['datenumeric i did', 2], ['week in feb', 4], ['day of rolling', 1], 
#               ['opinion dr said', 3], ['phase is scheduled', 5], ['column from the', 1],
#               ['precedent agreement datenumeric', 1], ['call he said', 2], ['tier thanks chris', 1], ['offering the process', 15]]

i = 2
while i < len(trainingData) - 3:
  next = trainingData[i+1]
  oneG = trainingData[i]
  twoG = trainingData[i]
  threeG = trainingData[i-1] + " " + trainingData[i]
  fourG = trainingData[i-2] + " " + trainingData[i-1] + " " + trainingData[i]

  #build one gram model
  if oneGramModel.get(oneG,-1) == -1:
    oneGramModel[oneG] = 1
  else:
    oneGramModel[oneG] += 1
  #build two gram model
  if twoGramModel.get(twoG,-1) == -1:
    twoGramModel[twoG] = {next:1}
  elif twoGramModel[twoG].get(next,-1) == -1:
    twoGramModel[twoG][next] = 1
  else:
    twoGramModel[twoG][next] += 1
  #build three gram model
  if threeGramModel.get(threeG,-1) == -1:
    threeGramModel[threeG] = {next:1}
  elif threeGramModel[threeG].get(next,-1) == -1:
    threeGramModel[threeG][next] = 1
  else:
    threeGramModel[threeG][next] += 1
  #build four gram model
  if fourGramModel.get(fourG,-1) == -1:
    fourGramModel[fourG] = {next:1}
  elif fourGramModel[fourG].get(next,-1) == -1:
    fourGramModel[fourG][next] = 1
  else:
    fourGramModel[fourG][next] += 1
  i+=1

#print oneGramModel
#print twoGramModel
#print threeGramModel
#print fourGramModel
print "...done"

#convert vals in dict to log2-space
print "converting models to -log2-space..."
subdictToLog2Space(oneGramModel)
print "one gram done"
dictToLog2Space(twoGramModel)
print "two gram done"
dictToLog2Space(threeGramModel)
print "three gram done"
dictToLog2Space(fourGramModel)
#print twoGramModel
print "...done"


#now test the models
lambdas = (0.08,0.45,0.30,0.17)  #generally these are decent lambdas values (model weights)
#lambdas = (0.25,0.25,0.25,0.25)  #generally these are decent lambdas values (model weights)
nGramScores = {"bool":0.0,"real":0.0,"top10":0.0,"pp":0.0,"recall":0.0,"avgwordlen":0.0}
vectorScores = {"bool":0.0,"real":0.0,"top10":0.0,"pp":0.0,"recall":0.0,"avgwordlen":0.0}
ensembleScores = {"bool":0.0,"real":0.0,"top10":0.0,"pp":-1.0,"recall":0.0,"avgwordlen":0.0}
correctWords = {}  #stores correct predictions for various analyses


i = 15
ntoks = len(testData)
while i < ntoks - 1 and i < 10000:
  
  ### ngram model prediction
  #generate a list of results as word|float pairs
  nGramResults = predictNgramModel(oneGramModel,twoGramModel,threeGramModel,fourGramModel,testData,i,lambdas)
  #print "given >"+testData[i-2]+" "+testData[i-1]+" "+testData[i]+"<\n",results
  if len(nGramResults) > 0:
    nGramResults.sort(compareLogs)
    '''
    #output testing: store the correctly predicted words for analysis
    if nGramResults[0][1] == testData[i+1]:
      print "hit!",nGramResults[0]
      if correctWords.get(nGramResults[0][1],-1) < 0:
        correctWords[nGramResults[0][1]] = 1
      else:
        correctWords[nGramResults[0][1]] += 1
    #print "\n\n",results
    '''
    nGramScores = scoreNgram(testData[i+1],nGramResults,nGramScores)

  ### vector model prediction
  vectorResults = predictVectorModel(vectorModel,coDistTable,testData,i,wordIdManager)
  #print "vector prediction for >"+testData[i]+"< :",vectorResults
  #raw_input()
  if len(vectorResults) > 0:
    #output testing: store the correctly predicted words for analysis
    if vectorResults[0][0].split(" ")[0] == testData[i+1]:
      print "vector hit!",vectorResults[0][0].split(" ")[0]
      if correctWords.get(vectorResults[0][0].split(" ")[0],-1) < 0:
        correctWords[vectorResults[0][0].split(" ")[0]] = 1
      else:
        correctWords[vectorResults[0][0].split(" ")[0]] += 1
    #score results
    vectorScores = scoreVector(testData[i+1],vectorResults,vectorScores)
    #printScores(scores,i)
  
  ### ensemble prediction, based off both ngrams and vector model predictions
  ensembleResults = predictEnsemble(nGramResults,vectorResults)
  if len(ensembleResults) > 0:
    ensembleScores = scoreEnsemble(testData[i+1],ensembleResults,ensembleScores)
  

  #loop update
  i+=1
  if (i % 200) == 0:
    print "ngram       >>"
    printScores_1(nGramScores,i)
    print "vectormodel >>"
    printScores_2(vectorScores,i)
    print "ensemble    >>"
    printScores_2(ensembleScores,i)

print "nTraining tokens="+str(len(trainingData))
print "nTest tokens="+str(len(testData))
print "lambdas: ",lambdas


printMostAccurateWords(correctWords)


'''
vector results more broadly distributed:
the:219
datenumeric:218
to:154
be:58
you:51
of:48
a:43
me:28
will:26
and:24
i:23
know:21
germany:18
up:17
as:16
are:16
is:15
that:15
following:14
with:14
for:14
in:13
on:13
if:12
have:11
at:11
robin:11
like:11
this:9
system:9
let:9
we:9
compression:8
deal:8
production:8
gathering:7
than:6
number:6
capacity:6
transmission:6
sure:5
dominion:5
out:5
desk:5
mail:5
charge:5
e:4
approved:4
after:4
week:4

'''




