"""Streamlit interface for the Turkish fact-check assistant."""

# imports and page setups
import pandas as pd
import streamlit as st

from src import config
from src.generation import fact_check, get_articles
from src.retrieval import get_collection, get_embedder

st.set_page_config(page_title="Teyit Asistanı", page_icon="🔎", layout="centered")

VERDICTS = {
    "TRUE": ("Doğru", "#1f8a4c"),
    "FALSE": ("Yanlış", "#c0392b"),
    "MIXED": ("Karma", "#d68910"),
    "NOT_ENOUGH_EVIDENCE": ("Yeterli kanıt yok", "#6c757d"),
}

EXAMPLES = [
    "Amazon teslim edilemeyen paketleri 2 euroya satıyormuş",
    "Threads hesabını silmek için Instagram hesabını da silmek gerekiyor",
    "Dolmalık biberlerin cinsiyeti varmış",
    "Ankara'da ekim ayında kar yağdı",
]

MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
          "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

def turkish_date(iso_date):
    """'2023-07-11' -> '11 Temmuz 2023'."""
    year, month, day = iso_date.split("-")
    return f"{int(day)} {MONTHS[int(month) - 1]} {year}"

# load the cache 
@st.cache_resource(show_spinner="Arşiv ve model yükleniyor...")
def load_resources():
    """Load the embedding model, vector store and articles once per server."""
    get_embedder()
    get_collection()
    return len(get_articles())

@st.cache_data
def load_metrics():
    """Read the evaluation results saved by notebook 07."""
    return pd.read_csv(config.RESULTS_DIR / "verdict_metrics.csv")

def verdict_badge(verdict):
    """Large coloured box showing the verdict."""
    label, color = VERDICTS[verdict]
    st.markdown(
        f"<div style='background:{color}; color:white; padding:0.7rem 1rem; "
        f"border-radius:0.6rem; font-size:1.4rem; font-weight:600; "
        f"text-align:center;'>{label}</div>",
        unsafe_allow_html=True,
    )

def evidence_card(number, item, used, show_summary=True):
    """One retrieved fact-check, with its source and whether it was used."""
    with st.container(border=True):
        st.markdown(f"**Kanıt {number}** · {item['source_name']} · {turkish_date(item['date'])}")
        st.markdown(f"*{item['claim']}*")

        left, right = st.columns(2)
        left.caption(f"Kuruluşun kararı: **{item['label']}**")
        right.caption("✓ Bu iddiayla eşleşti" if used else "Farklı bir iddia, karara katılmadı")

        if show_summary:
            with st.expander("Bulguların özeti"):
                st.write(item["summary"])
        st.link_button("Doğrulama yazısını aç", item["url"])

with st.sidebar:
    st.header("Nasıl çalışır?")
    st.markdown(
        "1. İddian, 2016–2023 arasında yayımlanmış doğrulama yazılarının arşivinde aranır.\n"
        "2. En ilgili üç yazı bulunur ve her birinin iddiayla **aynı** olup olmadığı değerlendirilir.\n"
        "3. Karar yalnızca aynı iddiayı inceleyen yazılara dayanır; yoksa *yeterli kanıt yok* denir."
    )

    st.header("Değerlendirme")
    metrics = load_metrics()
    rag, no_rag = metrics.iloc[0], metrics.iloc[1]
    st.metric(
        "Doğruluk (100 iddia)",
        f"{rag['accuracy']:.0%}",
        f"{(rag['accuracy'] - no_rag['accuracy']) * 100:+.0f} puan, kanıtsız modele göre",
    )
    st.metric("Ters karar", int(rag["opposite_verdicts"]),
              f"{int(no_rag['opposite_verdicts'])} kanıtsız modelde", delta_color="off")

    st.header("Sınırlar")
    st.caption(
        "Arşiv Temmuz 2023'te bitiyor; sonraki gelişmeler yansımayabilir. "
        "Görsel ve video içerikli iddialar arşivde az. "
        "Bu araç profesyonel doğrulamanın yerini tutmaz."
    )

n_articles = load_resources()

st.title("Teyit Asistanı")
n_text = f"{n_articles:,}".replace(",", ".")
st.caption(
    f"Teyit, Doğruluk Payı ve Doğrula'nın {n_text} doğrulama yazısından "
    "kanıt bularak Türkçe iddiaları değerlendirir."
)

if "claim" not in st.session_state:
    st.session_state.claim = ""

st.write("Bir örnek dene:")
example_cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if example_cols[i % 2].button(example, use_container_width=True):
        st.session_state.claim = example

claim = st.text_area(
    "Doğrulamak istediğin iddia",
    key="claim",
    height=100,
    placeholder="Örneğin: Bir bakanın patates firmasında danışmanlık yaptığı söyleniyor...",
)

if st.button("Doğrula", type="primary", use_container_width=True,
             disabled=not claim.strip()):
    with st.spinner("Arşiv taranıyor ve kanıtlar değerlendiriliyor..."):
        try:
            result, evidence = fact_check(claim)
        except RuntimeError as error:
            st.error(f"Şu anda cevap verilemiyor: {error}")
            st.stop()
    st.session_state.result = (claim, result, evidence)

if "result" in st.session_state:
    asked, result, evidence = st.session_state.result
    st.divider()
    st.caption(f"İddia: {asked}")
    verdict_badge(result.verdict)
    st.write("")
    st.write(result.explanation)

    used_ids = {a.evidence_id for a in result.evidence_assessments if a.is_same_claim}
    numbered = list(enumerate(evidence, start=1))
    used = [(n, item) for n, item in numbered if n in used_ids]
    others = [(n, item) for n, item in numbered if n not in used_ids]

    if used:
        st.subheader("Kararın dayandığı doğrulamalar")
        for number, item in used:
            evidence_card(number, item, used=True)
    else:
        st.info("Arşivde bu iddiayı inceleyen bir doğrulama bulunamadı.")

    if others:
        with st.expander(f"Aramada bulunan ama farklı iddiaları inceleyen yazılar ({len(others)})"):
            st.caption("Bu yazılar konu olarak benzer olduğu için bulundu, "
                       "ancak farklı bir iddiayı incelediği için karara katılmadı.")
            for number, item in others:
                evidence_card(number, item, used=False, show_summary=False)

st.divider()
st.caption("Veri: FCTR (Cekinel et al., 2024). Kararlar Gemini tarafından, "
           "doğrulama kuruluşlarının yazılarına dayanarak üretilir.")






