from typing import Dict
import streamlit as st
import functools
import wikipedia
from transformers import Pipeline
from transformers import pipeline

from config import config

def conditional_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if config["framework"] == "pt":
            qa = st.cache(func)(*args, **kwargs)
        else:
            qa = func(*args, **kwargs)
        return qa

    return wrapper

# NUM_SENT = 10

@conditional_decorator
@st.cache(allow_output_mutation=True)
def get_qa_pipeline() -> Pipeline:
    qa = pipeline('question-answering', framework=config["framework"])
    return qa

def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> Dict:
    input = {
        "question": question,
        "context": paragraph
    } 
    return pipeline(input)

@conditional_decorator
@st.cache(allow_output_mutation=True)
def get_wiki_paragraph(query: str) -> str:
    results = wikipedia.search(query)
    try:
        summary = wikipedia.summary(results[0], sentences=config["NUM_SENT"])
    except wikipedia.DisambiguationError as e:
        ambiguous_terms = e.options
        return wikipedia.summary(ambiguous_terms[0], sentences=config["NUM_SENT"])
    return summary

def format_text(paragraph: str, start_idx:int, end_idx: int) -> str:
    return ( 
        paragraph[:start_idx] 
        + "**" 
        + paragraph[start_idx:end_idx] 
        + "**" 
        + paragraph[end_idx:]
    )

def card(id_val, source, context):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title">{source}</h5>
            <h6 class="card-subtitle mb-2 text-muted">{id_val}</h6>
            <p class="card-text">{context}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    
    st.write("""
    # AI Questing and Answering with Streamlit
    """)

    
    paragraph_slot = st.empty()
    wiki_query = st.text_input("WIKIPEDIA SEARCH TERM", "")
    st.write("""
    # Aak Question from above Term
    """)
    question = st.text_input("QUESTION", "")

    if wiki_query:
        wiki_para = get_wiki_paragraph(wiki_query)
        paragraph_slot.markdown(wiki_para)
        if question != "":
            pipeline = get_qa_pipeline()
            # st.write(pipeline.model)
            # st.write(pipeline.model.config)
            try:
                answer = answer_question(pipeline, question, wiki_para)

                start_idx = answer["start"]
                end_idx = answer["end"]
                st.success(answer["answer"])
                print(f"NUM SENT: {config['NUM_SENT']}")
                print(f"FRAMEWORK: {config['framework']}")
                print(f"QUESTION: {question}\nRESPONSE: {answer}")
                paragraph_slot.markdown(format_text(wiki_para, start_idx, end_idx))
                
            except Exception as e:
                print(e)
                st.warning("You must provide a valid wikipedia paragraph")