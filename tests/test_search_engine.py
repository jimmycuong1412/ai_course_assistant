from pathlib import Path
import pytest
from search_engine import CourseSearchEngine


@pytest.fixture
def mock_resources_dir(tmp_path):
    """Creates a temporary mock directory structure with dummy files."""
    assignments_dir = tmp_path / "assignments"
    assignments_dir.mkdir()
    
    file_path = assignments_dir / "Assignment 04 - Tenacity.pdf"
    file_path.write_text("Dummy PDF content")
    return tmp_path


def test_determine_category():
    engine = CourseSearchEngine.__new__(CourseSearchEngine)
    
    assert engine._determine_category(Path("assignments/Assignment_01.pdf")) == "assignments"
    assert engine._determine_category(Path("workshops/Workshop_02.pdf")) == "workshops"
    assert engine._determine_category(Path("guidelines/Guide.pdf")) == "guidelines"
    assert engine._determine_category(Path("other/doc.pdf")) == "general"


def test_tokenize_and_lemmatize(mocker):
    # Mock spaCy model and internal indexing methods to speed up test execution
    mocker.patch.object(CourseSearchEngine, "_ingest_pdfs")
    mocker.patch.object(CourseSearchEngine, "_build_bm25_index")
    
    engine = CourseSearchEngine(Path("fake_dir"))
    
    # Mock spaCy token structure
    class DummyToken:
        def __init__(self, text, lemma, is_stop=False, is_alpha=True, like_num=False):
            self.text = text
            self.lemma_ = lemma
            self.is_stop = is_stop
            self.is_alpha = is_alpha
            self.like_num = like_num

    class DummyDoc:
        def __init__(self, tokens):
            self.tokens = tokens
        def __iter__(self):
            return iter(self.tokens)

    tokens = [
        DummyToken("running", "run"),
        DummyToken("tests", "test"),
        DummyToken("is", "be", is_stop=True),
        DummyToken("5", "5", is_alpha=False, like_num=True)
    ]
    engine.nlp = lambda text: DummyDoc(tokens)

    result = engine._tokenize_and_lemmatize("running tests is 5")
    assert result == ["run", "test", "5"]


def test_format_search_results():
    engine = CourseSearchEngine.__new__(CourseSearchEngine)
    
    empty_res = engine.format_search_results([])
    assert "No relevant course documents found" in empty_res

    mock_docs = [{
        "source_file": "Assignment 04.pdf",
        "page_number": 1,
        "category": "assignments",
        "content": "This is a test document."
    }]
    formatted = engine.format_search_results(mock_docs)
    assert "DOCUMENT 1" in formatted
    assert "Assignment 04.pdf" in formatted
    assert "This is a test document." in formatted