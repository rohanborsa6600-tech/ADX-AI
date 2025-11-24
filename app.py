import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# --- 1. Page Config ---
st.set_page_config(page_title="Smaran AI", page_icon="🤖")
st.title("📖 Smaran AI: Chat with PDF")

# --- 2. Sidebar for API Key ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Google API Key:", type="password")
    st.info("Get your free API Key here: [Google AI Studio](https://aistudio.google.com/app/apikey)")

# --- 3. Function to Process PDF ---
@st.cache_resource
def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text_chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            chunks = text.split('\n\n')
            for chunk in chunks:
                if len(chunk) > 30:
                    text_chunks.append(f"[Page {i+1}] {chunk}")
    
    # Train Model
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_chunks)
    nn = NearestNeighbors(n_neighbors=3, metric='cosine')
    nn.fit(X)
    
    return text_chunks, vectorizer, nn

# --- 4. Main Interface (File Uploader) ---
st.write("तुमची PDF फाईल खाली अपलोड करा:")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # फाईल प्रोसेस करा
    with st.spinner("PDF वाचत आहे... कृपया थांबा..."):
        try:
            corpus, vectorizer, nn = process_pdf(uploaded_file)
            st.success("✅ PDF यशस्वीरित्या वाचली! आता प्रश्न विचारा.")
            
            # --- Chat Interface ---
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("तुमचा प्रश्न विचारा..."):
                if not api_key:
                    st.warning("⚠️ कृपया डाव्या बाजूला API Key टाका!")
                    st.stop()

                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                # --- AI Logic ---
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')

                    q_vec = vectorizer.transform([prompt])
                    distances, indices = nn.kneighbors(q_vec)
                    context = "\n".join([corpus[i] for i in indices[0]])

                    full_prompt = f"""
                    You are a helpful assistant. Use the context below to answer the question in Marathi.
                    If the answer is not in the context, say "माहिती उपलब्ध नाही".
                    
                    Context: {context}
                    Question: {prompt}
                    """
                    
                    response = model.generate_content(full_prompt)
                    answer = response.text

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error: {e}")

        except Exception as e:
            st.error(f"PDF वाचताना एरर आला: {e}")
else:
    st.info("कृपया सुरुवात करण्यासाठी PDF अपलोड करा.")
