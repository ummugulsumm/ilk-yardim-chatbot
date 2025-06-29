import streamlit as st
import os
import re
from dotenv import load_dotenv
from rag.rag_pipeline import get_rag_pipeline, get_context
from models import FlanT5RAGModel, GeminiRAGModel

def remove_surrogates(text):
    """Surrogate karakterleri temizle"""
    return re.sub(r'[\ud800-\udfff]', '', text) if isinstance(text, str) else text


load_dotenv()


st.set_page_config(
    page_title="İlk Yardım Chatbot",
    page_icon="🚑",
    layout="wide"
)

PDF_PATH = os.path.join("data", "ilk-yardim.pdf")
CHROMA_DIR = "chroma_db"

st.sidebar.title("⚙️ Ayarlar")

use_flan_t5 = st.sidebar.checkbox("Flan-T5 Kullan", value=True)
use_gemini = st.sidebar.checkbox("Gemini Kullan", value=True)


k_context = st.sidebar.slider("Context Parça Sayısı", min_value=1, max_value=10, value=5)


gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    st.sidebar.success("✅ Gemini API Key bulundu")
else:
    st.sidebar.warning("⚠️ Gemini API Key bulunamadı")
    st.sidebar.info("💡 .env dosyasına GEMINI_API_KEY ekleyin")

@st.cache_resource
def check_database():
    """Vektör veritabanı durumunu kontrol et"""
    if os.path.exists(CHROMA_DIR):
        return True, "✅ Vektör veritabanı hazır"
    else:
        return False, "⚠️ Vektör veritabanı bulunamadı"

db_ready, db_status = check_database()
st.sidebar.info(db_status)

st.title("🚑 İlk Yardım Chatbot")
st.markdown("**Flan-T5** ve **Gemini** modellerini kullanarak ilk yardım sorularınızı yanıtlayın.")

if not db_ready:
    st.warning("""
    ⚠️ **Vektör veritabanı henüz oluşturulmamış!**
    
    İlk kullanımda vektör veritabanı oluşturulması gerekiyor. Bu işlem biraz zaman alabilir.
    
    **Hızlı kurulum için:**
    ```bash
    python setup_database.py
    ```
    """)

example_questions = [
    "Yanık durumunda ne yapmalıyım?",
    "Kalp krizi belirtileri nelerdir?",
    "Kanama durumunda nasıl müdahale edilir?",
    "Bayılma durumunda ne yapılmalı?",
    "Zehirlenme durumunda ilk yardım nasıl yapılır?"
]

selected_example = st.selectbox("Örnek sorular:", ["Kendi sorunuzu yazın"] + example_questions)

user_query = st.text_input("Soru:", placeholder="İlk yardım ile ilgili sorunuzu yazın...")

if selected_example != "Kendi sorunuzu yazın":
    user_query = selected_example

if st.button("🔍 Cevapla", type="primary"):
    if not user_query or not user_query.strip():
        st.error("Lütfen bir soru girin!")
    elif use_gemini and not gemini_api_key:
        st.error("Gemini kullanmak için .env dosyasına GEMINI_API_KEY ekleyin!")
    else:
        with st.spinner("📚 PDF'den bilgi aranıyor ve modeller çalıştırılıyor..."):
            try:
                # RAG pipeline'ı başlat
                pipeline = get_rag_pipeline(PDF_PATH, CHROMA_DIR)
                
                # Vektör veritabanı kontrolü
                if not db_ready:
                    st.info("🔄 Vektör veritabanı oluşturuluyor... (Bu işlem birkaç dakika sürebilir)")
                
                context = pipeline.get_context(user_query, k=k_context)
                
                results = {}
                
                # Flan-T5 modeli
                if use_flan_t5:
                    with st.spinner("Flan-T5 modeli çalışıyor..."):
                        try:
                            flan_model = FlanT5RAGModel()
                            flan_answer = flan_model.generate_answer(user_query, context)
                            results['flan_t5'] = {
                                'answer': flan_answer
                            }
                            st.success("✅ Flan-T5 tamamlandı")
                        except Exception as e:
                            st.error(f"❌ Flan-T5 hatası: {str(e)}")
                            results['flan_t5'] = {
                                'answer': f"Hata: {str(e)}"
                            }
                
                # Gemini modeli
                if use_gemini and gemini_api_key:
                    with st.spinner("Gemini modeli çalışıyor..."):
                        try:
                            gemini_model = GeminiRAGModel()  # API key otomatik olarak .env'den alınacak
                            gemini_answer = gemini_model.generate_answer(user_query, context)
                            results['gemini'] = {
                                'answer': gemini_answer
                            }
                            st.success("✅ Gemini tamamlandı")
                        except ValueError as e:
                            st.error(f"Gemini API hatası: {e}")
                            use_gemini = False
                        except Exception as e:
                            st.error(f"❌ Gemini hatası: {str(e)}")
                            results['gemini'] = {
                                'answer': f"Hata: {str(e)}"
                            }
                
                if results:
                    st.success("✅ Analiz tamamlandı!")
                    
                    with st.expander("📖 Kullanılan Bilgi Kaynakları"):
                        st.text(remove_surrogates(context))
                    
                    if len(results) > 1:
                        cols = st.columns(len(results))
                        for i, (model_name, result) in enumerate(results.items()):
                            with cols[i]:
                                st.markdown(f"### {'🤖 Flan-T5' if model_name == 'flan_t5' else '🌟 Gemini'}")
                                st.markdown(f"**💬 Cevap:**\n{remove_surrogates(result['answer'])}")
                    else:
                        model_name, result = list(results.items())[0]
                        st.markdown(f"### {'🤖 Flan-T5' if model_name == 'flan_t5' else '🌟 Gemini'} Sonucu")
                        st.markdown(f"**💬 Cevap:**\n{remove_surrogates(result['answer'])}")
                else:
                    st.warning("⚠️ Hiçbir model çalıştırılamadı!")
                    
            except Exception as e:
                st.error(f"❌ Hata oluştu: {str(e)}")
                st.info("💡 Hata detayları için console'u kontrol edin.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚑 İlk Yardım Chatbot</p>
    <p>Bu uygulama ilk yardım PDF'inden bilgi çıkararak sorularınızı yanıtlar.</p>
</div>
""", unsafe_allow_html=True) 