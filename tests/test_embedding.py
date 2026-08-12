import json
import sqlite3
import pytest
import sqlite_vec
from src.util.nlp import get_question_skeleton, get_few_shot

# ---------------------------------------------------------------------------
# Shared test schema (simplified_ddl JSON format used throughout the pipeline)
# ---------------------------------------------------------------------------

SCHEMA = json.dumps([
    "department(department_id INTEGER, name TEXT, budget REAL)",
    "employee(employee_id INTEGER, name TEXT, department_id INTEGER, salary REAL)",
])


# Real simplified_ddl format: no type annotations, plural table names,
# compound column names with underscores.
SCHEMA_REAL = json.dumps([
    "Students(student_id, student_details)",
    "Courses(course_id, course_name, course_description)",
    "Addresses(address_id, line_1, city, zip_postcode, country)",
    "People(person_id, first_name, last_name, email_address)",
    "Student_Course_Registrations(student_id, course_id, registration_date)",
    "Student_Course_Attendance(student_id, course_id, date_of_attendance)",
])


# ---------------------------------------------------------------------------
# get_question_skeleton
# ---------------------------------------------------------------------------

class TestGetQuestionSkeleton:
    def test_masks_table_names(self):
        skeleton = get_question_skeleton("List all departments", SCHEMA)
        assert "department" not in skeleton.lower()
        assert "<mask>" in skeleton

    def test_masks_column_names(self):
        skeleton = get_question_skeleton("What is the budget of each department?", SCHEMA)
        assert "budget" not in skeleton.lower()
        assert "<mask>" in skeleton

    def test_masks_numeric_values(self):
        skeleton = get_question_skeleton("Find employees with salary greater than 50000", SCHEMA)
        assert "50000" not in skeleton
        assert "<mask>" in skeleton

    def test_masks_quoted_strings(self):
        skeleton = get_question_skeleton("Find the department named 'Engineering'", SCHEMA)
        assert "Engineering" not in skeleton
        assert "<mask>" in skeleton

    def test_preserves_non_domain_words(self):
        skeleton = get_question_skeleton("How many employees are there?", SCHEMA)
        # "how", "many", "are", "there" are not domain tokens
        assert "how" in skeleton.lower()
        assert "many" in skeleton.lower()

    def test_empty_schema_returns_question(self):
        # Invalid JSON schema → fall back to original question
        question = "How many rows?"
        result = get_question_skeleton(question, "not valid json")
        assert result == question

    def test_returns_string(self):
        result = get_question_skeleton("Show all employees", SCHEMA)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# get_few_shot (requires embedding DB and gold DB)
# ---------------------------------------------------------------------------

class TestGetFewShot:
    def test_returns_list(self, index_db, gold_db):
        result = get_few_shot("How many departments are there?", SCHEMA, index_db, gold_db)
        assert isinstance(result, list)

    def test_returns_nonempty_result(self, index_db, gold_db):
        result = get_few_shot("How many employees work in each department?", SCHEMA, index_db, gold_db)
        assert len(result) > 0

    def test_result_contains_sql(self, index_db, gold_db):
        result = get_few_shot("What is the average salary?", SCHEMA, index_db, gold_db)
        all_sql = " ".join(ex["sql"] for ex in result)
        assert any(kw in all_sql.upper() for kw in ("SELECT", "FROM", "WHERE", "COUNT", "AVG"))

    def test_result_has_expected_fields(self, index_db, gold_db):
        result = get_few_shot("List all departments with more than 10 employees", SCHEMA, index_db, gold_db, top_k=3)
        assert len(result) >= 1
        assert {"question", "sql", "distance"} <= result[0].keys()

    def test_top_k_respected(self, index_db, gold_db):
        result_1 = get_few_shot("How many employees?", SCHEMA, index_db, gold_db, top_k=1)
        result_3 = get_few_shot("How many employees?", SCHEMA, index_db, gold_db, top_k=3)
        assert len(result_3) >= len(result_1)

    def test_distance_is_float(self, index_db, gold_db):
        result = get_few_shot("How many employees?", SCHEMA, index_db, gold_db, top_k=1)
        assert isinstance(result[0]["distance"], float)

    def test_missing_index_raises_on_bad_path(self, gold_db):
        import pytest, sqlite3
        with pytest.raises((FileNotFoundError, OSError, sqlite3.OperationalError)):
            get_few_shot("Any question", SCHEMA, "/nonexistent/path/embedding.sqlite", gold_db)


# ---------------------------------------------------------------------------
# embedding_dataset schema (requires embedding DB)
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {"vector", "id", "db_id", "source", "question", "skeleton_question"}


def _open(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


class TestEmbeddingSchema:
    def test_table_exists(self, index_db):
        conn = _open(index_db)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_dataset'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_all_columns_present(self, index_db):
        conn = _open(index_db)
        # vec0 virtual tables expose column info via PRAGMA table_info
        cols = {row[1] for row in conn.execute("PRAGMA table_info(embedding_dataset)")}
        conn.close()
        assert EXPECTED_COLUMNS <= cols

    def test_rows_are_nonempty(self, index_db):
        conn = _open(index_db)
        count = conn.execute("SELECT count(*) FROM embedding_dataset").fetchone()[0]
        conn.close()
        assert count > 0

    def test_question_column_populated(self, index_db):
        conn = _open(index_db)
        row = conn.execute(
            "SELECT question FROM embedding_dataset WHERE question IS NOT NULL LIMIT 1"
        ).fetchone()
        conn.close()
        assert row is not None and row[0]

    def test_skeleton_question_column_populated(self, index_db):
        conn = _open(index_db)
        row = conn.execute(
            "SELECT skeleton_question FROM embedding_dataset WHERE skeleton_question IS NOT NULL LIMIT 1"
        ).fetchone()
        conn.close()
        assert row is not None and row[0]


# ---------------------------------------------------------------------------
# get_question_skeleton — known masking gaps (currently failing)
#
# Root causes:
#   1. Plural table names: schema has "Students" → domain token is "students".
#      Question uses singular "student" → lemmatize("student") = "student" ≠ "students".
#   2. Compound column names: schema has "student_id" → domain token is "student_id".
#      Question uses "id" or "course" alone — these subwords are never in domain_tokens.
#   3. Multi-part compound: "course_name" → domain token is "course_name".
#      Question word "name" / "names" never matches "course_name".
# ---------------------------------------------------------------------------

class TestGetQuestionSkeletonKnownGaps:
    # --- Bug 1: plural table name vs singular question token ---

    def test_singular_masked_when_table_is_plural(self):
        # Table "Students" → domain token "students".
        # Question token "student" → lemmatize("student") = "student" ≠ "students" → not masked.
        result = get_question_skeleton("find the student with the highest grade", SCHEMA_REAL)
        assert "student" not in result.lower(), (
            "singular 'student' should be masked when table name is 'Students'"
        )

    def test_plural_question_token_masked_when_table_is_plural(self):
        # "students" in question → lemmatize("students") = "student" ≠ "students" → not masked.
        result = get_question_skeleton("how many students are registered for each course?", SCHEMA_REAL)
        assert "students" not in result.lower(), (
            "'students' in question should be masked when table name is 'Students'"
        )

    def test_singular_course_masked_when_table_is_plural(self):
        # Table "Courses" → domain token "courses".
        # Question token "course" → lemmatize("course") = "course" ≠ "courses" → not masked.
        result = get_question_skeleton("which course has the most registrations?", SCHEMA_REAL)
        assert "course" not in result.lower(), (
            "singular 'course' should be masked when table name is 'Courses'"
        )

    # --- Bug 2: constituent word of a compound column name ---

    def test_id_masked_from_compound_column(self):
        # Column "student_id" → domain token "student_id".
        # Question token "id" → not in domain_tokens → not masked.
        result = get_question_skeleton("list the id of every student", SCHEMA_REAL)
        tokens = result.split()
        assert "<mask>" in tokens and "id" not in tokens, (
            "'id' should be masked when 'student_id' is a column name"
        )

    def test_ids_masked_from_compound_column(self):
        # "ids" → lemmatize("ids") → likely "ids" (WordNet may not reduce this) → not masked.
        result = get_question_skeleton("what are the ids of students who never attended a course?", SCHEMA_REAL)
        assert "ids" not in result.lower(), (
            "'ids' should be masked when compound columns like 'student_id' are present"
        )

    def test_address_masked_from_compound_column(self):
        # Column "email_address" → domain token "email_address".
        # Question token "address" → not in domain_tokens → not masked.
        result = get_question_skeleton("find the address of each person", SCHEMA_REAL)
        assert "address" not in result.lower(), (
            "'address' should be masked when 'email_address' is a column name"
        )

    # --- Bug 3: partial word of multi-part compound column name ---

    def test_name_masked_from_course_name_column(self):
        # Column "course_name" → domain token "course_name".
        # Question token "name" → not in domain_tokens → not masked.
        result = get_question_skeleton("what is the name of each course?", SCHEMA_REAL)
        assert "name" not in result.lower(), (
            "'name' should be masked when 'course_name' is a column name"
        )

    def test_names_masked_from_compound_column(self):
        # "names" → lemmatize → "name", still not in domain_tokens containing "course_name".
        result = get_question_skeleton("list the first and last names of all people", SCHEMA_REAL)
        assert "names" not in result.lower(), (
            "'names' should be masked when compound columns like 'first_name'/'last_name' are present"
        )
