from .tokenizer import KoreanTokenizer
from collections import Counter

class WordVectorEncoder:
    """
    토큰화된 단어들을 숫자 벡터로 변환합니다.
    먼저 단어들을 고유 ID에 매핑하는 어휘집(vocabulary)을 구축합니다.
    """
    def __init__(self, tokenizer: KoreanTokenizer = None):
        self.tokenizer = tokenizer if tokenizer is not None else KoreanTokenizer()
        self.word_to_id = {}
        self.id_to_word = {}

    def build_vocab(self, corpus: list[str], min_freq: int = 2):
        """
        주어진 텍스트 모음(corpus)으로부터 어휘집을 구축합니다.
        min_freq: 어휘집에 포함될 최소 단어 빈도수
        """
        # 1. 전체 코퍼스를 토큰화하고 단어 빈도를 계산합니다.
        words = []
        for text in corpus:
            tokens = self.tokenizer.get_significant_tokens(text)
            words.extend(tokens)
        
        word_counts = Counter(words)

        # 2. 특수 토큰 추가: <PAD> (패딩), <UNK> (알 수 없는 단어)
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>'}

        # 3. 최소 빈도수 이상의 단어만 어휘집에 추가합니다.
        next_id = 2
        for word, count in word_counts.items():
            if count >= min_freq:
                self.word_to_id[word] = next_id
                self.id_to_word[next_id] = word
                next_id += 1
        
        print(f"어휘집 구축 완료. 총 단어 수: {len(self.word_to_id)}")

    def encode(self, text: str, max_length: int = None) -> list[int]:
        """
        하나의 문장을 숫자 ID의 시퀀스로 변환합니다.
        max_length가 지정되면, 시퀀스의 길이를 맞추기 위해 패딩 또는 잘라내기를 수행합니다.
        """
        # 1. 문장을 주요 품사 토큰으로 분리합니다.
        tokens = self.tokenizer.get_significant_tokens(text)

        # 2. 각 토큰을 어휘집의 ID로 변환합니다.
        unk_id = self.word_to_id.get('<UNK>')
        encoded_ids = [self.word_to_id.get(token, unk_id) for token in tokens]

        # 3. max_length에 맞춰 패딩 또는 잘라내기를 적용합니다.
        if max_length is not None:
            pad_id = self.word_to_id.get('<PAD>')
            if len(encoded_ids) < max_length:
                encoded_ids += [pad_id] * (max_length - len(encoded_ids))
            else:
                encoded_ids = encoded_ids[:max_length]
        
        return encoded_ids

    def decode(self, encoded_ids: list[int]) -> list[str]:
        """
        숫자 ID 시퀀스를 단어의 리스트로 복원합니다.
        """
        # 0번 ID(<PAD>)는 무시하고 단어로 변환합니다.
        pad_id = self.word_to_id.get('<PAD>')
        decoded_words = [self.id_to_word.get(id) for id in encoded_ids if id != pad_id]
        return decoded_words

if __name__ == '__main__':
    # 간단한 테스트
    corpus_texts = [
        "안녕하세요, 엘런 프로젝트의 한국어 토크나이저 테스트입니다.",
        "두 번째 문장입니다. 아 마스터듀얼 하고싶다",
        "세 번째 문장, 살려주세요."
    ]

    encoder = WordVectorEncoder()
    encoder.build_vocab(corpus_texts, min_freq=1) # 테스트를 위해 모든 단어 포함

    print("Word to ID:", encoder.word_to_id)
    print("ID to Word:", encoder.id_to_word)

    print("\n--- 인코딩 및 패딩 테스트 ---")
    test_sentence = "엘런 프로젝트는 새로운 프로젝트입니다."
    encoded_result = encoder.encode(test_sentence, max_length=10)
    print(f"원본 문장: {test_sentence}")
    print(f"인코딩 및 패딩 결과 (max_length=10): {encoded_result}")

    print("\n--- 디코딩 테스트 ---")
    decoded_words = encoder.decode(encoded_result)
    print(f"디코딩 결과: {decoded_words}")
