"""
test_models.py — Dataclass & Pydantic v2 tests
"""

import pytest
from pydantic import ValidationError

from day1.models import (
    Coordinate,
    Course,
    DataPoint,
    Dataset,
    DatasetStats,
    ExperienceLevel,
    MLExperiment,
    ModelMetrics,
    Student,
    TrainingRequest,
)


# ─────────────────────────────────────────────
# DATACLASS TESTS
# ─────────────────────────────────────────────

def test_student_creation(sample_student):
    assert sample_student.name == "Alice"
    assert sample_student.age == 25
    assert sample_student.email == "alice@example.com"
    assert sample_student.gpa == 3.5


def test_student_is_passing(sample_student):
    assert sample_student.is_passing() is True


def test_student_failing_gpa():
    s = Student("Bob", 20, "bob@test.com", gpa=1.5)
    assert s.is_passing() is False


def test_frozen_dataclass_raises_on_mutation():
    c = Coordinate(1.0, 2.0)
    with pytest.raises(Exception):   # FrozenInstanceError
        c.x = 99.0                   # type: ignore


def test_coordinate_distance():
    c1 = Coordinate(0.0, 0.0)
    c2 = Coordinate(3.0, 4.0)
    assert c1.distance_to(c2) == pytest.approx(5.0)


def test_coordinate_distance_symmetric():
    c1, c2 = Coordinate(1.0, 2.0), Coordinate(4.0, 6.0)
    assert c1.distance_to(c2) == pytest.approx(c2.distance_to(c1))


def test_course_enrollment_within_limit(sample_course):
    assert sample_course.enroll("Alice") is True
    assert sample_course.enroll("Bob")   is True
    assert sample_course.enroll("Carol") is True     # hits max (3)
    assert sample_course.available_seats == 0


def test_course_enrollment_rejects_over_limit(sample_course):
    sample_course.enroll("A")
    sample_course.enroll("B")
    sample_course.enroll("C")
    assert sample_course.enroll("D") is False        # over capacity


def test_ml_experiment_level_auto_advanced():
    exp = MLExperiment("run1", "RF", accuracy=0.95)
    assert exp.level == ExperienceLevel.ADVANCED


def test_ml_experiment_level_auto_intermediate():
    exp = MLExperiment("run2", "LR", accuracy=0.75)
    assert exp.level == ExperienceLevel.INTERMEDIATE


def test_ml_experiment_invalid_accuracy_raises():
    with pytest.raises(ValueError, match="accuracy"):
        MLExperiment("bad", "SVM", accuracy=1.5)


def test_dataset_add_and_size(xor_dataset):
    assert xor_dataset.size == 4


def test_dataset_labels(xor_dataset):
    assert sorted(set(xor_dataset.labels)) == ["F", "T"]


def test_datapoint_dimension():
    dp = DataPoint([1.0, 2.0, 3.0], "A")
    assert dp.dimension == 3


# ─────────────────────────────────────────────
# PYDANTIC TESTS
# ─────────────────────────────────────────────

def test_pydantic_valid_request(sample_training_request):
    assert sample_training_request.model_name == "iris_clf"
    assert sample_training_request.algorithm == "random_forest"
    assert sample_training_request.test_size == pytest.approx(0.2)


def test_pydantic_invalid_algorithm_raises():
    with pytest.raises(ValidationError) as exc:
        TrainingRequest(
            model_name="m", algorithm="neural_net",
            features=["f1"], target="y",
        )
    assert "algorithm" in str(exc.value).lower()


def test_pydantic_invalid_test_size_raises():
    with pytest.raises(ValidationError):
        TrainingRequest(
            model_name="m", algorithm="svm",
            features=["f1"], target="y",
            test_size=0.9,               # > 0.5 → fails ge/le constraint
        )


def test_pydantic_model_dump(sample_training_request):
    d = sample_training_request.model_dump()
    assert isinstance(d, dict)
    assert d["features"] == ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def test_pydantic_model_dump_json(sample_training_request):
    j = sample_training_request.model_dump_json()
    assert "iris_clf" in j
    assert "random_forest" in j


def test_dataset_stats_valid():
    stats = DatasetStats(
        num_samples=150, num_features=4, num_classes=3,
        class_distribution={"setosa": 50, "versicolor": 50, "virginica": 50},
    )
    assert stats.num_samples == 150
    assert stats.num_classes == 3


def test_dataset_stats_distribution_mismatch_raises():
    with pytest.raises(ValidationError):
        DatasetStats(
            num_samples=100, num_features=4,
            class_distribution={"A": 60, "B": 30},   # 90 ≠ 100
        )


def test_model_metrics_is_classification():
    m = ModelMetrics(accuracy=0.92, precision=0.91, recall=0.90, f1_score=0.91)
    assert m.is_classification is True


def test_model_metrics_is_regression():
    m = ModelMetrics(mse=4.5)
    assert m.is_classification is False
