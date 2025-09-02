# 🤖💬 Hybrid-FAQ-Chatbot: Semantic + Lexical Retrieval with LLM Refinement  

![Chatbot Screenshot](static/images/interface.png)  
![Hybrid Retrieval Flow](static/images/architecture.png)  

---

## 🔍 Overview  
Most FAQ chatbots rely on rigid keyword search, which fails when users phrase questions differently (synonyms, paraphrases, or informal queries).  

This project introduces an **intelligent hybrid chatbot** that combines:  
- 🧠 **Semantic Retrieval** with FAISS + Sentence-BERT embeddings  
- 🔑 **Lexical Retrieval** with BM25 keyword ranking  
- 🔄 **Paraphrase Augmentation** using T5-based rephraser  
- ✨ **LLM Response Refinement** with Together API (Qwen / Llama models)  

The result is a system that retrieves **accurate answers** while generating **natural, personalized responses**.  

---

## ⚙️ Features  
- 📌 **Hybrid Search**: Combines semantic + lexical retrieval for robustness  
- 📝 **Keyword Extraction**: Enhanced extraction with SpaCy, KeyBERT, and WordNet synonyms  
- 🔄 **Data Expansion**: Augments 60 FAQ entries into 240 via paraphrasing  
- ⚡ **Fast Retrieval**: < 60 ms response time with FAISS indexing  
- 🎨 **Interactive UI**: Built with Streamlit for real-time use  
- 💾 **Session Management**: Saves user history with JSON persistence  
- 🤝 **Generative Refinement**: Produces conversational responses tailored to the user  

---

## 🏗️ Architecture  
```mermaid
flowchart TD
    A[User Query] --> B{Hybrid Search}
    B -->|Semantic| C[FAISS + SBERT]
    B -->|Lexical| D[BM25]
    C --> E[Candidate Answers]
    D --> E
    E --> F[LLM Refinement (Together API)]
    F --> G[Final Response]


📂 Project Structure
├── data/                   # FAQ dataset + augmented paraphrases  
├── keyword_extraction.py   # Enhanced keyword extractor (SpaCy + KeyBERT + WordNet)  
├── paraphrase_expansion.py # Question rephraser (T5-based)  
├── app.py                  # Streamlit chatbot application  
├── newdata.json            # Expanded FAQ data  
├── faiss_index/            # Vector index storage  
├── requirements.txt        # Dependencies  
└── README.md               # Project documentation  




🚀 Installation
Clone the repository

bash
Copy code

🚀 Installation
Clone the repository
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Set environment variables
Create a .env file:
TOGETHER_API_KEY=your_api_key_here

▶️ Running the Chatbot

streamlit run app.py

