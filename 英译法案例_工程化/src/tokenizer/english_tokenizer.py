from nltk import TreebankWordTokenizer, TreebankWordDetokenizer
from src.tokenizer.base_tokenizer import BaseTokenizer

class EnglishTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    # 改造为类方法
    @classmethod
    def tokenize(cls,text) -> list[str]:
        return cls.tokenizer.tokenize(text)

    def decode(self,indexes) -> str:
        tokens = [self.index2word[index] for index in indexes]
        return self.detokenizer.detokenize(tokens)

if __name__ == '__main__':
    pass