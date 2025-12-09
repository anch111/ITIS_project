# src/main.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)


class PlagiarismDetector:

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        # Приводим к нижнему регистру
        text = text.lower()
        # Удаляем знаки препинания
        text = re.sub(r'[^\w\s]', '', text)
        # Токенизация
        tokens = nltk.word_tokenize(text)
        # Удаление стоп-слов и лемматизация
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens
                  if word not in self.stop_words]
        return ' '.join(tokens)

    def calculate_cosine_similarity(self, texts):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix

    def longest_common_subsequence(self, s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def ngram_similarity(self, s1, s2, n=3):
        def get_ngrams(text, n):
            return [text[i:i+n] for i in range(len(text)-n+1)]

        ngrams1 = set(get_ngrams(s1, n))
        ngrams2 = set(get_ngrams(s2, n))

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union

    def detect_plagiarism(self, texts):
        if len(texts) < 2:
            return []

        # Предобработка всех текстов
        processed_texts = [self.preprocess_text(text)
                           for text in texts]

        # Расчёт сходства по трём методам
        cosine_sim = self.calculate_cosine_similarity(processed_texts)
        results = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                # Среднее сходство по трём методам
                lcs_score = self.longest_common_subsequence(
                    processed_texts[i],
                    processed_texts[j])
                # Нормализуем LCS (по максимальной длине)
                max_len = max(len(processed_texts[i]),
                              len(processed_texts[j]))
                lcs_norm = lcs_score / max_len if max_len > 0 else 0

                ngram_score = self.ngram_similarity(processed_texts[i],
                                                    processed_texts[j])

                # Комбинируем (можно веса изменить)
                combined_score = \
                    (cosine_sim[i][j] + lcs_norm + ngram_score) / 3

                if combined_score >= self.threshold:
                    results.append((i, j, combined_score))

        return results
