import jpype
from jpype.types import *
import os
import sys

class KoreanTokenizer:
    def __init__(self):
        """
        생성자: 필요한 모든 .jar 파일을 클래스패스에 추가하여 JVM을 시작합니다.
        """
        if not jpype.isJVMStarted():
            # 프로젝트 루트를 안정적으로 찾습니다.
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while not os.path.isdir(os.path.join(project_root, '.git')):
                parent_dir = os.path.dirname(project_root)
                if parent_dir == project_root:
                    raise FileNotFoundError("프로젝트 루트(.git 폴더)를 찾을 수 없습니다.")
                project_root = parent_dir

            # Using Open Korean Text (Apache-2.0) for Korean NLP processing
            jar_path = os.path.join(project_root, "src", "external", "okt", "KoreanTextProcessingUtilitiesSBT-assembly-4.0.jar")

            if not os.path.exists(jar_path):
                raise FileNotFoundError(f"Fat Jar 파일을 찾을 수 없습니다: {jar_path}")

            # JPype에 단일 Fat Jar 파일을 리스트로 전달합니다.
            jpype.startJVM(
                jpype.getDefaultJVMPath(),
                classpath=[jar_path],
                convertStrings=True
            )

    def tokenize(self, text: str) -> list[tuple[str, str]]:
        """
        문장을 입력받아 형태소 분석을 수행합니다.
        최신 API는 정적(static) 메소드를 직접 호출하는 방식입니다.
        """
        # Java 클래스를 가져옵니다.
        OktJava = JClass("org.openkoreantext.processor.OpenKoreanTextProcessorJava")
        
        # 정적 메소드인 tokenize를 직접 호출합니다.
        # 이 메소드는 Java의 List<KoreanToken>을 반환합니다.
        java_tokens = OktJava.tokenize(text)

        # Scala의 연결 리스트를 수동으로 순회하여 Python 리스트로 변환합니다.
        python_tokens_intermediate = []
        while not java_tokens.isEmpty():
            python_tokens_intermediate.append(java_tokens.head())
            java_tokens = java_tokens.tail()

        # 변환된 Python 리스트를 사용하여 최종 결과를 만듭니다.
        python_tokens = [(token.text(), token.pos().toString()) for token in python_tokens_intermediate]
        return python_tokens

    def get_nouns(self, text: str) -> list[str]:
        """
        문장에서 명사만 추출하여 리스트로 반환합니다.
        """
        tokens = self.tokenize(text)
        nouns = [word for word, pos in tokens if pos == 'Noun']
        return nouns

    def get_significant_tokens(self, text: str) -> list[str]:
        """
        문장에서 주요 품사(명사, 동사, 형용사)의 단어만 추출하여 리스트로 반환합니다.
        """
        tokens = self.tokenize(text)
        significant_words = [word for word, pos in tokens if pos in ['Noun', 'Verb', 'Adjective']]
        return significant_words

# --- 이하 테스트 코드는 동일 --- (생략)
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    try:
        tokenizer = KoreanTokenizer()
        text = "앙 기모띠. 와 샌즈 와 파피루스! 달리다. 달렸다. 달리겠습니다. 안녕하세요! 엉덩이가 없어진 것을 축하드립니다."
        
        print("--- 기본 토크나이징 (형태소 분석) ---")
        tokens = tokenizer.tokenize(text)
        print(tokens)

    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
