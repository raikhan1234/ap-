queries = []

def store_query(question, answer):
    queries.append({"question": question, "answer": answer})

def get_all_queries():
    return queries

def clear_all_queries():
    global queries
    queries = []
