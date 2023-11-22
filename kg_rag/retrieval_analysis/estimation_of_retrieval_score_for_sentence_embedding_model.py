import pandas as pd
import numpy as np
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
import sys
import time


VECTOR_DB_PATH = sys.argv[1]
SENTENCE_EMBEDDING_MODEL = sys.argv[2]
QUESTION_DATA_PATH = sys.argv[3]
SAVE_DATA_PATH = sys.argv[4]

SAVE_RETRIEVAL_SCORE_PATH = os.path.join(SAVE_DATA_PATH, "retrieval_score_for_test_questions_using_{}.csv".format("_".join(SENTENCE_EMBEDDING_MODEL.split("/")[-1].split("-"))))
MAX_SEARCH = 10000

def main():
    start_time = time.time()
    question_data = pd.read_csv(QUESTION_DATA_PATH)
    embedding_function = SentenceTransformerEmbeddings(model_name=SENTENCE_EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, 
                         embedding_function=embedding_function)
    total_questions = question_data.shape[0]
    retrieval_score_list = []
    for question_id in range(total_questions):
        question = question_data.questions.values[question_id]
        question_in_vectorDB = question_data.questions_as_in_database.values[question_id]
        search_result = vectorstore.similarity_search_with_score(question, k=MAX_SEARCH)
        for index, item in enumerate(search_result):
            if question_in_vectorDB in item[0].page_content:
                break
        score_range = (search_result[-1][-1]-search_result[0][-1])/(search_result[-1][-1]+search_result[0][-1])
        question_score = (search_result[-1][-1]-search_result[index][-1])/(search_result[-1][-1]+search_result[index][-1])
        retrieval_score = np.divide(question_score, score_range)
        retrieval_score_list.append((question, retrieval_score, index+1))
    question_retrieval_score_df = pd.DataFrame(retrieval_score_list, columns=["question", "retrieval_score", "documents_retrieved"])
    question_retrieval_score_df.to_csv(SAVE_RETRIEVAL_SCORE_PATH, index=False, header=True)
    end_time = round((time.time() - start_time)/(60*60), 2)
    print("Retrieval score estimation for {} completed in {} hrs".format(SENTENCE_EMBEDDING_MODEL, end_time))

if __name__ == "__main__":
    main()


