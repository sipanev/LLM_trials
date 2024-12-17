import json
from Tokenizer import *

class TokenCollector:
    def __init__(self):
        self.mTokensSet = set()

    def GetTokensSet(self):
        return self.mTokensSet

    def AddTokensFromTextFile(self, filename):
        file = open(filename, "r", encoding="utf8")
        lines = file.readlines()
        for line in lines:
            tokens = Tokenizer.TextToTokens(line, True)
            for word in tokens:
                self.mTokensSet.add(word)

    def SaveTokens(self, filename):
        tokensList = list(self.mTokensSet)
        tokensList.insert(0, '__ZER0__')
        with open(filename, 'w') as outfile:
            json.dump(tokensList, outfile)
