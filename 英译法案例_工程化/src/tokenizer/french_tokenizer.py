from nltk import word_tokenize
from src.tokenizer.base_tokenizer import BaseTokenizer

class FrenchDetokenizer:
    """法语专用 detokenizer，遵循法语标点空格规范。

    法语规则:
      - ? ! : ; » 前面加空格
      - « 后面加空格
      - . , ... ) ] } 紧贴前面的词（不留空格）
      - ( [ { 紧贴后面的词
    """
    # 前面需要空格的标点
    SPACE_BEFORE = set('?!:;»')
    # 后面需要空格的标点
    SPACE_AFTER = set('«')
    # 紧贴前面的标点
    ATTACH_LEFT = set('.,)]}') | {'...'}
    # 紧贴后面的标点
    ATTACH_RIGHT = set('([{')

    def detokenize(self, tokens: list[str]) -> str:
        if not tokens:
            return ''
        parts = [tokens[0]]
        for token in tokens[1:]:
            if token in self.SPACE_BEFORE:
                parts.append(' ' + token)
            elif token in self.ATTACH_LEFT:
                parts.append(token)
            elif parts and parts[-1] in self.ATTACH_RIGHT:
                parts.append(token)
            elif parts and parts[-1] in self.SPACE_AFTER:
                parts.append(' ' + token)
            else:
                parts.append(' ' + token)
        return ''.join(parts)

class FrenchTokenizer(BaseTokenizer):
    detokenizer = FrenchDetokenizer()

    @classmethod
    def tokenize(cls, text) -> list[str]:
        return word_tokenize(text, language='french')

    def decode(self, indexes) -> str:
        tokens = [self.index2word[index] for index in indexes]
        return self.detokenizer.detokenize(tokens)

if __name__ == '__main__':
    pass