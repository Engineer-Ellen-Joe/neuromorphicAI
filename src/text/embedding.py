import numpy as np

class EmbeddingLayer:
    """
    단어 ID를 의미를 담은 밀집 벡터(dense vector)로 변환하는 임베딩 계층.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        임베딩 계층을 초기화합니다.

        :param vocab_size: 어휘집의 총 단어 수
        :param embedding_dim: 각 단어를 표현할 벡터의 차원
        """
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("vocab_size와 embedding_dim은 0보다 커야 합니다.")
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # 임베딩 행렬을 무작위 값으로 초기화합니다.
        # 이 값들은 모델 학습 과정에서 점차 의미있는 값으로 조정됩니다.
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.01

        # 패딩 토큰(ID 0)의 벡터는 항상 0으로 유지합니다.
        self.embedding_matrix[0] = np.zeros(embedding_dim)

    def forward(self, id_sequence: list[int]) -> np.ndarray:
        """
        숫자 ID 시퀀스를 임베딩 벡터의 시퀀스로 변환합니다. (순전파)
        """
        # 입력 시퀀스의 각 ID에 해당하는 벡터를 임베딩 행렬에서 조회합니다.
        embedded_sequence = self.embedding_matrix[id_sequence]
        return embedded_sequence

if __name__ == '__main__':
    # 이전 단계에서 만든 WordVectorEncoder를 가져와서 테스트합니다.
    from .encoder import WordVectorEncoder

    # 1. 어휘집 구축
    corpus_texts = [
        "안녕하세요, 엘런 프로젝트의 한국어 토크나이저 테스트입니다.",
        "두 번째 문장입니다. 아 마스터듀얼 하고싶다",
        "세 번째 문장, 살려주세요."
    ]
    encoder = WordVectorEncoder()
    encoder.build_vocab(corpus_texts, min_freq=1)

    # 2. 문장 인코딩 및 패딩
    test_sentence = "엘런 프로젝트는 새로운 프로젝트입니다."
    encoded_padded = encoder.encode(test_sentence, max_length=10)

    # 3. 임베딩 계층 생성 및 순전파 테스트
    vocab_size = len(encoder.word_to_id)
    embedding_dimension = 8 # 각 단어를 8차원 벡터로 표현

    embedding_layer = EmbeddingLayer(vocab_size, embedding_dimension)
    vector_sequence = embedding_layer.forward(encoded_padded)

    print(f"--- 임베딩 테스트 ---")
    print(f"어휘집 크기: {vocab_size}")
    print(f"임베딩 차원: {embedding_dimension}")
    print(f"\n인코딩된 ID 시퀀스 (패딩 포함):\n{encoded_padded}")
    print(f"\n임베딩 벡터 시퀀스 (shape: {vector_sequence.shape}):\n{vector_sequence}")
