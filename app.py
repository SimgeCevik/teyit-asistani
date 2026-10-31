"""Streamlit interface for the Turkish fact-check assistant."""

# imports and page setups
import pandas as pd
import requests
import streamlit as st

from src import config
from src.generation import fact_check, fact_check_live, get_articles
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
    """'2023-07-11' -> '11 Temmuz 2023'; empty dates are shown as unknown."""
    if not iso_date:
        return "tarih belirtilmemiş"
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
    verdicts = pd.read_csv(config.RESULTS_DIR / "verdict_metrics.csv")
    abstention = pd.read_csv(config.RESULTS_DIR / "abstention_metrics.csv")
    return verdicts, abstention

def verdict_badge(verdict):
    """Large coloured box showing the verdict."""
    label, color = VERDICTS[verdict]
    st.markdown(
        f"<div style='background:{color}; color:white; padding:0.7rem 1rem; "
        f"border-radius:0.6rem; font-size:1.4rem; font-weight:600; "
        f"text-align:center;'>{label}</div>",
        unsafe_allow_html=True,
    )

def verdict_pill(verdict):
    """Small coloured label, used in the sidebar legend."""
    label, color = VERDICTS[verdict]
    return (f"<span style='display:inline-block; background:{color}; color:white; "
            f"padding:0.1rem 0.6rem; margin:0 0.3rem 0.4rem 0; border-radius:1rem; "
            f"font-size:0.8rem;'>{label}</span>")

def evidence_card(number, item, used, show_summary=True, summary_label="Bulguların özeti"):
    """One retrieved fact-check, with its source and whether it was used."""
    with st.container(border=True):
        st.markdown(f"**Kanıt {number}** · {item['source_name']} · {turkish_date(item['date'])}")
        st.markdown(f"*{item['claim']}*")

        left, right = st.columns(2)
        left.caption(f"Kuruluşun kararı: **{item['label']}**")
        right.caption("✓ Bu iddiayla eşleşti" if used else "Farklı bir iddia, karara katılmadı")

        if show_summary:
            with st.expander(summary_label):
                st.write(item["summary"])
        st.link_button("Doğrulama yazısını aç", item["url"])

def show_evidence(result, evidence, summary_label="Bulguların özeti"):
    """Matched evidence as cards; unmatched results in a collapsed section."""
    used_ids = {a.evidence_id for a in result.evidence_assessments if a.is_same_claim}
    numbered = list(enumerate(evidence, start=1))
    used = [(n, item) for n, item in numbered if n in used_ids]
    others = [(n, item) for n, item in numbered if n not in used_ids]

    if used:
        st.subheader("Kararın dayandığı doğrulamalar")
        for number, item in used:
            evidence_card(number, item, used=True, summary_label=summary_label)

    if others:
        with st.expander(f"Aramada bulunan ama farklı iddiaları inceleyen yazılar ({len(others)})"):
            st.caption("Bu yazılar konu olarak benzer olduğu için bulundu, "
                       "ancak farklı bir iddiayı incelediği için karara katılmadı.")
            for number, item in others:
                evidence_card(number, item, used=False, show_summary=False)

with st.sidebar:
    st.header("Nasıl çalışır?")
    st.markdown(
        "1. İddian, 2016–2023 arasında yayımlanmış **3.238 doğrulama yazısının** "
        "arşivinde aranır.\n"
        "2. Bulunan her yazının iddianla **aynı iddiayı** inceleyip incelemediği "
        "değerlendirilir; karar yalnızca bunlara dayanır.\n"
        "3. Arşivde karşılık yoksa, aynı kuruluşların **son yayımlanan doğrulamaları** "
        "Google Fact Check üzerinden aranır.\n"
        "4. İkisinde de bulunamazsa *yeterli kanıt yok* denir; sistem tahmin yürütmez."
    )

    st.divider()
    st.header("Karar türleri")
    st.markdown("".join(verdict_pill(v) for v in VERDICTS), unsafe_allow_html=True)
    st.caption("Karma: İddianın bir kısmı doğru, bir kısmı yanlış.")

    st.divider()
    st.header("Kaynaklar")
    st.markdown(
        "**Arşiv:** Teyit, Doğruluk Payı ve Doğrula yazıları (FCTR veri seti, 2016–2023)\n\n"
        "**Güncel:** Aynı üç kuruluşun Google Fact Check'te yayımlanan doğrulamaları"
    )

    st.divider()
    st.header("Değerlendirme")
    st.caption("Arşiv katmanı, 100 iddialık dengeli test seti")
    verdicts, abstention = load_metrics()
    rag, no_rag = verdicts.iloc[0], verdicts.iloc[1]
    st.metric(
        "Doğruluk",
        f"{rag['accuracy']:.0%}",
        f"{(rag['accuracy'] - no_rag['accuracy']) * 100:+.0f} puan, kanıtsız modele göre",
    )
    
    st.metric("Ters karar", int(rag["opposite_verdicts"]))
    st.caption(f"Kanıtsız modelde: {int(no_rag['opposite_verdicts'])}")

    st.metric("Arşivde olmayan iddiada uydurmama", f"{abstention.loc[0, 'adjusted_correct']:.0%}")
    st.caption(
        f"Katı ölçümde {abstention.loc[0, 'correct_abstention']:.0%}; aradaki fark, "
        "aynı iddianın arşivde başka bir kuruluşça da doğrulanmış olmasından kaynaklanıyor."
    )

    st.divider()
    st.header("Sınırlar")
    st.caption(
        "Arşiv Temmuz 2023'te bitiyor. Güncel katman yalnızca bu üç kuruluşun daha önce "
        "doğruladığı iddiaları bulabilir; bu yazıların yalnızca başlığını ve kararını görür, "
        "içeriğini göremez. Güncel katman değerlendirme sonuçlarına dahil değildir. "
        "Video ve görsel içerikli iddialar yalnızca metin olarak değerlendirilir. "
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

    live = (None, [], None)
    if result.verdict == "NOT_ENOUGH_EVIDENCE":
        with st.spinner("Güncel doğrulamalar aranıyor..."):
            try:
                live_result, live_evidence = fact_check_live(claim)
                live = (live_result, live_evidence, None)
            except (requests.RequestException, RuntimeError) as error:
                live = (None, [], str(error))

    st.session_state.result = (claim, result, evidence)
    st.session_state.live = live

if "result" in st.session_state:
    asked, result, evidence = st.session_state.result
    live_result, live_evidence, live_error = st.session_state.get("live", (None, [], None))

    live_decided = live_result is not None and live_result.verdict != "NOT_ENOUGH_EVIDENCE"
    if live_decided:
        final, final_evidence = live_result, live_evidence
        source_note = "Güncel doğrulamalara göre (Google Fact Check, son yayımlanan doğrulamalar)"
    else:
        final, final_evidence = result, evidence
        source_note = "Doğrulama arşivine göre (2016–2023)"

    # 1. The answer
    st.divider()
    st.caption(f"İddia: {asked}")
    verdict_badge(final.verdict)
    st.caption(source_note)
    st.write(final.explanation)
    show_evidence(final, final_evidence, summary_label="Doğrulama başlığı" if live_decided else "Bulguların özeti")

    # 2. How we got there
    if live_decided:
        with st.expander("Arşiv taraması: Bu iddia 2016–2023 arşivinde bulunamadı"):
            st.write(result.explanation)
            st.caption("Arşivde bulunan en yakın yazılar:")
            for number, item in enumerate(evidence, start=1):
                evidence_card(number, item, used=False, show_summary=False)

    elif result.verdict == "NOT_ENOUGH_EVIDENCE":
        if live_error:
            st.warning(f"Güncel doğrulamalar şu anda alınamadı: {live_error}")
        else:
            st.info("Arşivde ve güncel doğrulamalarda bu iddiayı inceleyen bir yazı bulunamadı. "
                    "İddia henüz doğrulanmamış olabilir.")
        if live_evidence:
            with st.expander(f"Güncel aramada bulunan ama farklı iddiaları inceleyen yazılar "
                             f"({len(live_evidence)})"):
                st.caption("Bu yazılar iddianla birebir aynı değil, ancak konuyla ilgili bilgi "
                           "içerebilir. Güncel aramada yalnızca yazıların başlığı görülebildiği için "
                           "ayrıntılar için doğrulama yazısını açmanı öneririz.")
                for number, item in enumerate(live_evidence, start=1):
                    evidence_card(number, item, used=False, show_summary=False)

st.divider()
st.caption("Veri: FCTR (Cekinel et al., 2024). Kararlar Gemini tarafından, "
           "doğrulama kuruluşlarının yazılarına dayanarak üretilir.")






