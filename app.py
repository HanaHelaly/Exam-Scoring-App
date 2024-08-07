import pandas as pd
import time
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.load import dumps, loads
from groq import Groq
import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from datasets import Dataset
import io
import chardet
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def setup_vectorstore(uploaded_files):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    text_chunks = text_splitter.split_documents(uploaded_files[0]['model_answer'])
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore.as_retriever()


def extract_queries(text, question, num_queries=2):
    pattern = r'\#\s*(.*?\?)'
    text = text.strip()
    matches = re.finditer(pattern, text, re.DOTALL)
    questions = [match.group(1) for match in matches]
    return questions[:num_queries] + [question]


def retrieve_documents(question, retriever):
    retrieval_template = """
    You are an AI language model assistant. Your task is to generate two different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines.
    Original question: {question}
    Output (2 queries starting with '#'):
    """


    def get_groq_response(prompt_text):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            model="llama3-70b-8192",
            temperature=0.0

        )
        return chat_completion.choices[0].message.content
    prompt = retrieval_template.format(question=question)
    queries = get_groq_response(prompt)
    generate_queries = extract_queries(queries, question)

    retrieval_chain = ((lambda x: generate_queries) | retriever.map() | reciprocal_rank_fusion)
    docs = retrieval_chain.invoke(question)
    return docs


def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results


def extract_final_score(text):
    # Regular expression pattern to find the final score
    pattern = r'\%\%(\d+(?:\.\d+)?)\d*\/?\d*\%\%'

    match = re.search(pattern, text)

    if match:
        return float(match.group(1))
    else:
        return 1
def process_answer(reference_docs, question, answer):
    template = """
   **Instruction:**
    You are a student scoring exam system that uses a reference document to evaluate student answers. Your only information is the provided reference documents. Extract the correct answer from the documents.
    
    **Scoring Criteria:**
    
    * Evaluate the student's answer based on the following criteria:
        * Accuracy (out of 5 points): How well the student's answer correctly addresses the question.
        * Relevance (out of 2 points): How well the student's answer stays relevant to the question asked.
        * Completeness (out of 3 points): How well the student's answer covers essential aspects of the question. It does not have to cover all aspects of the question.
    
    * Final Score (out of 10 points): Add the accuracy, relevance, and completeness scores. If the student's answer is unavailable, missing, or not answering the question, GIVE ZERO IN ALL CATEGORIES.
    * Mention an explanation for the chosen score.
    
    Example 1:
    
    **Reference Document:**
    Newton’s laws of motion, three statements describing the relations between the forces acting on a body and the
    motion of the body, first formulated by English physicist and mathematician Isaac Newton,
    which are the foundation of classical mechanics. 
    The first law states that an object at rest will remain at rest, 
    and an object in motion will remain in motion unless acted upon by an external force. 
    The second law relates the acceleration of an object to the force applied to it and its mass. 
    The third law states that for every action, there is an equal and opposite reaction.
    
    **Question:**
    What are Newton's laws of motion?
    
    **Student's Answer:**
    Newton's laws of motion are three fundamental principles formulated by Sir Isaac Newton. The first law states that an object at rest will remain at rest, and an object in motion will remain in motion unless acted upon by an external force. The second law relates the acceleration of an object to the force applied to it and its mass. The third law states that for every action, there is an equal and opposite reaction.
    
    **Output:**
    Accuracy (out of 5 points): 5 points
    Relevance (out of 2 points): 2 points
    Completeness (out of 3 points): 3 points
    **Final Score: %%10%%**
    
    Example 2:
    
    **Reference Document:**
    Newton’s laws of motion, three statements describing the relations between the forces acting on a body and the
    motion of the body, first formulated by English physicist and mathematician Isaac Newton,
    which are the foundation of classical mechanics. 
    The first law states that an object at rest will remain at rest, 
    and an object in motion will remain in motion unless acted upon by an external force. 
    The second law relates the acceleration of an object to the force applied to it and its mass. 
    The third law states that for every action, there is an equal and opposite reaction.
    
    **Question:**
    What are Newton's laws of motion?
    
    **Student's Answer:**
    Newton's laws of motion are principles that explain how objects move under forces.
    
    **Output:**
    Accuracy (out of 5 points): 1 point
    Relevance (out of 2 points): 2 point
    Completeness (out of 3 points): 0 points
    **Final Score: %%3%%**
    
    Now, evaluate the following:
    
    **Reference Document:**
    {reference}
    
    **Student's Answer:**
    {answer}
    
    **Question:**
    {question}
    
    **Output:**
    Wrap the final score between double percentage signs. For example, if the final score is 8, output: %%8%%
    """


    def get_groq_response(prompt_text):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            model="llama3-70b-8192",
            temperature=0.0,
            top_p=0.1
        )
        return chat_completion.choices[0].message.content

    full_prompt = template.format(reference=reference_docs, answer=answer, question=question)

    groq_response = get_groq_response(full_prompt)
    print('Question:',question)
    print('Response',groq_response)
    print("--------------------------------")

    result = {
        "reference": reference_docs,
        "answer": answer,
        "question": question,
        "groq_response": groq_response
    }

    score = extract_final_score(result['groq_response'])
    result['score'] = score

    return result


def main():
    st.set_page_config(page_title="Test Corrector Model")
    col1, col2, col3 = st.columns([0.2, 5.6, 0.2])
    with col2:
        st.title("📝 Examination Test Scoring")
        st.markdown("###### Upload Questions Source PDF and Students' Answers CSV and Get Grades in a Few Minutes")
        st.text("")
        st.text("")

        uploaded_model_answer_pdf = st.file_uploader("Upload Model Answer PDF", type="pdf")
        model_answer = []
        if uploaded_model_answer_pdf:
            reader = PdfReader(uploaded_model_answer_pdf)
            i = 1
            for page in reader.pages:
                model_answer.append(Document(page_content=page.extract_text(), metadata={'page': i}))
                i += 1

        student_answer_csv = st.file_uploader("Upload Student Answer CSV", type="csv")
        if student_answer_csv:
            file_content = student_answer_csv.read()
            # Detect the encoding of the file
            result = chardet.detect(file_content)
            encoding = result['encoding']
            file_content = io.StringIO(file_content.decode(encoding))
            df = pd.read_csv(file_content)
            df.columns = df.columns.str.strip().str.lower()
            question_keywords = ["what", "how", "illustrate", "mention",
                                 "who", "when", "where", "why",
                                 "describe", "explain", "compare", "contrast",
                                 "define", "outline", "summarize", "discuss"]

            # Extract questions based on specific keywords and ending with '?'
            questions = [
                col for col in df.columns
                if any(col.lower().startswith(keyword) for keyword in question_keywords) or col.endswith('?')
            ]
            fullname_column = [col for col in df.columns if 'fullname' in col.replace(' ', '').lower()][0]

            structured_data = []
            for index, row in df.iterrows():
                name = row[fullname_column]
                if pd.isna(name):
                    name = 'No Name'
                for question in questions:
                    answer = row[question]
                    structured_data.append({
                        'name': name,
                        'question': question,
                        'answer': answer
                    })
            structured_df = pd.DataFrame(structured_data)
            dataset = Dataset.from_pandas(structured_df)

        if uploaded_model_answer_pdf and student_answer_csv:
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked = False

            col4, col5, col6 = st.columns([0.25, 5.5, 0.25])
            with col5:
                button_placeholder = st.empty()
                if not st.session_state.button_clicked:
                    if button_placeholder.button("Grade Answers"):
                        st.text("")
                        with st.spinner("Fetching Reference Documents..."):
                            st.session_state.button_clicked = True
                            button_placeholder.empty()
                            uploaded_files = [{"model_answer": model_answer}]
                            retriever = setup_vectorstore(uploaded_files)
                            reference_docs_dict = {}

                            for question in questions:
                                reference_docs_dict[question] = retrieve_documents(question, retriever)

                        results = []
                        with st.spinner("Grading Answers..."):
                            for student in dataset:
                                question = student['question']
                                answer = student['answer']
                                reference_docs = reference_docs_dict[question]
                                success = False
                                while not success:
                                    try:
                                        print(student['name'])
                                        result = process_answer(reference_docs=reference_docs, question=question, answer=answer)

                                        success = True
                                    except Exception as e:
                                        if 'rate limit' in str(e).lower():
                                            print(e)
                                            time.sleep(22)
                                    question_score = min(result['score'], 10)
                                    results.append({
                                        'student_name': student['name'],
                                        'question_score': question_score
                                    })

                        results_df = pd.DataFrame(results)
                        results_df['question_number'] = results_df.groupby('student_name').cumcount() + 1
                        pivot_df = results_df.pivot(index='student_name', columns='question_number', values='question_score')
                        pivot_df.columns = [f'Q{col}' for col in pivot_df.columns]
                        pivot_df.reset_index(inplace=True)
                        pivot_df['final_score'] = pivot_df.iloc[:, 1:].mean(axis=1) * 5
                        pivot_df['final_score'] = pivot_df['final_score'].round(1)
                        pivot_df = pivot_df.rename(columns={'final_score': 'Final Score (50)'})
                        threshold = 0.95 * len(pivot_df)
                        pivot_df.dropna(axis=1, thresh=threshold, inplace=True)

                        st.dataframe(pivot_df)
                        st.download_button(
                            label="Download Results",
                            data=pivot_df.to_csv().encode('utf-8'),
                            file_name="results.csv",
                            mime="text/csv",
                        )
                        st.session_state.button_clicked = False

if __name__ == "__main__":
    main()

