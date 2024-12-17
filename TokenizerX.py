import json
from Tokenizer import *

class TokenizerX:
    def __init__(self, filename):
        with open(filename, 'r') as infile:
            self.mTokens = json.loads(infile.read())
        self.mTokenLookup = {}
        for i in range(len(self.mTokens)):
            self.mTokenLookup[self.mTokens[i]] = i

    def GetTokenList(self):
        return self.mTokens

    def TextToNumbers(self, text, lowerCase):
        res = []
        ttokens = Tokenizer.TextToTokens(text, lowerCase)
        for tok in ttokens:
            if tok in self.mTokenLookup.keys():
                index = self.mTokenLookup[tok]
            else:
                index = 0
            res.append(index)
        return res

    def NumberToToken(self, index):
        if index < 0 or index >= len(self.mTokens):
            return '_INVALID_'
        return self.mTokens[index]