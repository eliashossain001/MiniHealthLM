from presidio_analyzer import AnalyzerEngine
import re

def sanitize_text(text):
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text, language='en')
    spans = [(r.start, r.end) for r in results]
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + "[REDACTED]" + text[end:]
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    return text

if __name__ == "__main__":
    with open("data/corpus.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    clean_text = sanitize_text(raw_text)
    with open("data/corpus.txt", "w", encoding="utf-8") as f:
        f.write(clean_text)
