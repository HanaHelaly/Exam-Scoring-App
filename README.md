# AI Exam Grading Tool using RAG and LLaMA3

This web application leverages **Retrieval-Augmented Generation (RAG)** and **LLaMA3**, allowing educators to automate the grading process for essay-based exams. The tool takes in the answer key provided by the teacher in a PDF format and the Google Forms responses from students, and it outputs a graded results sheet.

## Features

- **Automated Grading**: Uses RAG and LLaMA3 to assess student answers based on the teacher-provided answer key.
- **File Upload**: Teachers upload a PDF containing the official answers for the essay exam.
- **Student Responses**: Accepts Google Sheets with student answers, where the columns include the student's full name and their responses to each question.
- **Customizable Grading**: Flexible enough to handle various types of essay questions in various fields.
- **Graded Output**: Produces a final Google Sheet listing each student's name and their corresponding grade.

## How It Works

For the application, the system takes an input file, which is a CSV file outputted from Google Forms in a spreadsheet format, facilitating easy integration and use, it also takes the questions source in PDF file format. 

Input Files Format 

PDF FILE: Textbooks, or educational resources. 

CSV FILE: Student answers (google forms output - sheets). 

The following steps outline how to use the application: 

Navigate Link: [EXAM SCORING APP](https://examination-form-scoring.streamlit.app/)

1. Upload Files: Begin by uploading the PDF of the model answer and the CSV file containing students'
answers from your local PC.
2. Process Evaluation: Press the 'Grade Answers' button and wait a few minutes for the system to evaluate
the student answers.

3. Display or Download Results: Once the evaluation is complete, you can either display the results in a
table dataframe on the screen or download the evaluated results as a CSV file using the provided
download button.

## File Formats

- **Answer Key**: PDF format containing the teacher's book containing correct answers.
- **Student Responses**: Google Sheets format. Ensure the sheet contains:
  - Fullname column of the student.
  - other student details such as National ID (optional)
  - Subsequent columns: One column per essay question.
  

## Example

### Input Files:
- **Answer Key (PDF)**: `exam_answers.pdf`
- **Student Responses (Google Sheet)**: `student_responses.csv`

### Output:
- **Graded Results**: `graded_results.csv`

| Full Name       | Q1 Score | Q2 Score | Q3 Score | Total Score (out of 50)|
|-----------------|----------|----------|----------|-------------|
| Yasmine Ahmed      | 8        | 7        | 9        | 24          |
| Sarah  Mohamed    | 9        | 9        | 8        | 26          |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-exam-grading-tool.git
