from TokenizerX import *

tok = TokenizerX('tokens.txt')

print(len(tok.GetTokenList()),'tokens loaded')

res = tok.TestToNumbers('Awakened joke admit charitable butterflies, the worried lion sleeps at bed tonight.', True)
print(res)