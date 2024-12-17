from TokenCollector import *


folder = 'k:/ProjectData/texts'
filename = folder + '/TheGiftOfTheMagi.txt'

tokenizer = TokenCollector()

tokenizer.AddTokensFromTextFile(filename)

tokenList = list(tokenizer.GetTokensSet())

# Save them
tokenizer.SaveTokens('tokens.txt')

print(tokenList)
print(len(tokenList), 'tokens')

