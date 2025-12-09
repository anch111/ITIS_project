# tests/test_main.py

from src.main import PlagiarismDetector

def test_preprocess_text():
    detector = PlagiarismDetector()
    text = "Machine learning is a subset of artificial intelligence!"
    result = detector.preprocess_text(text)
    words = result.split()
    
    assert "machine" in words
    assert "learning" in words
    assert "subset" in words
    assert "artificial" in words
    assert "intelligence" in words

    assert "a" not in words
    assert "is" not in words
    assert "of" not in words

def test_cosine_similarity():
    detector = PlagiarismDetector()
    texts = [
        "Machine learning is a part of artificial intelligence.",
        "Machine learning is a key part of artificial intelligence."
    ]
    processed = [detector.preprocess_text(t) for t in texts]
    sim_matrix = detector.calculate_cosine_similarity(processed)
    similarity = float(sim_matrix[0][1])
    assert similarity > 0.84  # реалистичное значение

def test_lcs():
    detector = PlagiarismDetector()
    s1 = "machine learning model"
    s2 = "machine learning system"
    score = detector.longest_common_subsequence(s1, s2)
    assert score >= len("machine learning ")

def test_ngram_similarity():
    detector = PlagiarismDetector()
    s1 = "neural network deep"
    s2 = "neural network deep"
    score = detector.ngram_similarity(s1, s2, n=2)
    assert score == 1.0

def test_detect_plagiarism():
    detector = PlagiarismDetector(threshold=0.75)
    texts = [
        "Neural networks are widely used in deep learning models.",
        "Neural networks are commonly used in deep learning models.",
        "Thermodynamics studies heat and energy transfer."
    ]
    results = detector.detect_plagiarism(texts)
    assert len(results) == 1
    i, j, score = results[0]
    assert i == 0 and j == 1
    assert score >= 0.75  