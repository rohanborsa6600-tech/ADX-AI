import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os

# --- 1. वेबसाईटचे सेटिंग ---
st.set_page_config(page_title="Smaran AI", page_icon="🤖")
st.title("📖 Smaran AI: Chat with PDF")

# --- 2. Sidebar मध्ये API Key घेणे ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Google API Key:", type="password")
    st.info("तुमची मोफत API Key [येथे मिळवा](https://aistudio.google.com/app/apikey)")

# --- 3. फंक्शन: PDF वाचणे आणि मॉडेल बनवणे ---
@st.cache_resource
def load_data_and_model(pdf_file_path):
    # PDF वाचणे
    reader = PdfReader(pdf_file_path)
    text_chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            chunks = text.split('\n\n')
            for chunk in chunks:
                if len(chunk) > 30:
                    text_chunks.append(f"[Page {i+1}] {chunk}")
    
    # सर्च मॉडेल बनवणे
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_chunks)
    nn = NearestNeighbors(n_neighbors=3, metric='cosine')
    nn.fit(X)
    
    return text_chunks, vectorizer, nn

# --- 4. मुख्य प्रोसेस ---
pdf_filename = "Smaranpath-DH.pdf"  # तुमच्या PDF चे नाव तंतोतंत हेच असावे

if not os.path.exists(pdf_filename):
    st.error(f"⚠️ '{pdf_filename}' ही फाईल सापडली नाही. कृपया GitHub वर अपलोड करा.")
else:
    # डेटा लोड करा
    corpus, vectorizer, nn = load_data_and_model(pdf_filename)

    # चॅट इंटरफेस
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # जुन्या गप्पा दाखवा
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # नवीन प्रश्न विचारा
    if prompt := st.chat_input("तुमचा प्रश्न विचारा..."):
        if not api_key:
            st.warning("कृपया आधी Sidebar मध्ये API Key टाका!")
            st.stop()

        # युजरचा प्रश्न दाखवा
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # --- AI कडून उत्तर मिळवणे ---
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            # संदर्भ शोधणे
            q_vec = vectorizer.transform([prompt])
            distances, indices = nn.kneighbors(q_vec)
            context = "\n".join([corpus[i] for i in indices[0]])

            # AI ला प्रॉम्प्ट
            full_prompt = f"""
            You are a helpful assistant. Use the context below to answer the question in Marathi.
            If the answer is not in the context, say "माहिती उपलब्ध नाही".
            
            Context: {context}
            Question: {prompt}
            """
            
            response = model.generate_content(full_prompt)
            answer = response.text

            # उत्तर दाखवा
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error: {e}")
