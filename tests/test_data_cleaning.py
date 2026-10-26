"""Unit tests for the text cleaning utilities. """

from src.data_cleaning import clean_text, is_headline_style, split_glued_text

def test_splits_glued_sentences():
    assert split_glued_text("iddia edildi.İddialara göre") == "iddia edildi. İddialara göre"

def test_splits_glued_headings():
    assert split_glued_text("yazı eklenmişGörseldeki") == "yazı eklenmiş. Görseldeki"

def test_protected_words_survive():
    assert split_glued_text("WhatsApp grubunda yayıldı") == "WhatsApp grubunda yayıldı"

def test_text_without_protected_words_is_returned():
    assert split_glued_text("sade bir metin") == "sade bir metin"

def test_removes_related_links():
    text = "Bu iddia doğru değil. İlginizi çekebilir: Başka bir yazı"
    assert "İlginizi" not in clean_text(text)

def test_removes_related_links_in_the_middle():
    text = "Doğru değil. İlginizi çekebilir: Başka bir yazıKonuyla ilgili açıklama yapıldı."
    cleaned = clean_text(text)
    assert "İlginizi" not in cleaned
    assert "Konuyla ilgili açıklama yapıldı." in cleaned

def test_headline_detection():
    assert is_headline_style("Cocostar'da beyazlatılmış havuç yok")
    assert not is_headline_style("Amazon'un ucuz paket sattığı iddiası")
    assert not is_headline_style("Bir hesap paylaştı ve iddia edildi. Gönderi beğeni aldı.")
    assert not is_headline_style("Video gerçek mi?")

def test_removes_escaped_quotes():
    assert clean_text('önünde Hayır\\" dedi') == 'önünde Hayır" dedi'

def test_splits_sentence_starting_with_protected_word():
    assert split_glued_text("haberi paylaştı.WhatsApp üzerinden") == "haberi paylaştı. WhatsApp üzerinden"