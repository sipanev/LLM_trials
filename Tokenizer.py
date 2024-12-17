
class Tokenizer:
    mSpecialChars = ".,;:!?()’“”"

    @staticmethod
    def TextToTokens(text, lowerCase=False):
        fline = text
        # Capitalization
        if lowerCase:
            fline = fline.lower()
        # Some special processing
        for ch in list(Tokenizer.mSpecialChars):
            fline = fline.replace(ch, ' '+ch+' ')
        fline = fline.replace('  ', ' ')
        words = fline.split()
        return words


    #def TextToNumbers(self, text):
