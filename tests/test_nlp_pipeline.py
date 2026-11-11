import unittest
from nlp_pipeline import (
    normalize_entities,
    replace_homoglyphs,
    clean_and_tokenize_words,
    filter_lexicon,
    HOMOGLYPH_MAP,
)


class TestNLPPipeline(unittest.TestCase):
    def test_normalize_entities_url(self):
        text = "Visit our portal at https://github.com/youcef-ach or www.example.org for details."
        normalized = normalize_entities(text)
        self.assertNotIn("https://github.com", normalized)
        self.assertNotIn("www.example.org", normalized)
        self.assertEqual(normalized.count("<url>"), 2)

    def test_normalize_entities_email(self):
        text = "Send queries to yi.achouri@esi-sba.dz or contact@spooky.org."
        normalized = normalize_entities(text)
        self.assertNotIn("yi.achouri@esi-sba.dz", normalized)
        self.assertEqual(normalized.count("<email>"), 2)

    def test_replace_homoglyphs(self):
        obfuscated = "Th!$ !s c00l"
        cleaned = replace_homoglyphs(obfuscated)
        self.assertEqual(cleaned, "This is cool")

    def test_clean_and_tokenize_words(self):
        raw = "Check $p00ky text at http://spooky.ai from user@esi.dz!"
        tokens = clean_and_tokenize_words(raw)
        self.assertIn("spooky", tokens)
        self.assertIn("<url>", tokens)
        self.assertIn("<email>", tokens)

    def test_filter_lexicon(self):
        positive_lexicon = {"magnificent", "splendid", "wonder", "radiant"}
        sample_tokens = ["the", "mansion", "was", "magnificent", "and", "splendid"]
        matched = filter_lexicon(sample_tokens, positive_lexicon)
        self.assertEqual(matched, ["magnificent", "splendid"])


if __name__ == "__main__":
    unittest.main()
