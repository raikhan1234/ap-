from src.db_utils import store_query, get_all_queries, clear_all_queries

def test_store_and_get_queries():
    store_query("Test question", "Test answer")
    history = get_all_queries()
    assert len(history) == 1
    assert history[0]["question"] == "Test question"

def test_clear_queries():
    clear_all_queries()
    assert len(get_all_queries()) == 0
