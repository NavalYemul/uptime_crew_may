"""
models.py — Dataclasses & Pydantic v2
======================================
Java → Python transfer: POJO → @dataclass, Bean Validation → Pydantic.

Run:  python -m day1.models
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


# ══════════════════════════════════════════════════════
# PART 1 — DATACLASSES  (5 examples)
# Java equivalent: Lombok @Data / @Value / Java 16 record
# ══════════════════════════════════════════════════════

# ── Example 1: Basic mutable dataclass ───────────────
@dataclass
class Student:
    """
    Mutable POJO-style record.
    Java: public class Student { } with Lombok @Data
    """
    name: str
    age: int
    email: str
    gpa: float = 0.0

    def is_passing(self) -> bool:
        return self.gpa >= 2.0

    def __str__(self) -> str:
        return f"Student({self.name}, GPA={self.gpa})"


# ── Example 2: Frozen (immutable) dataclass ──────────
@dataclass(frozen=True)
class Coordinate:
    """
    Immutable value object.
    Java: record Coordinate(double x, double y) {}   (Java 16+)
    frozen=True → FrozenInstanceError on mutation (like final fields)
    """
    x: float
    y: float

    def distance_to(self, other: "Coordinate") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


# ── Example 3: Dataclass with default_factory ────────
@dataclass
class Course:
    """
    Mutable dataclass with a list field.
    CRITICAL: never use `enrolled: list = []` — that's a shared mutable default!
    Use field(default_factory=list) instead.  Same bug exists in Java btw.
    """
    course_id: str
    title: str
    max_students: int = 30
    enrolled: list[str] = field(default_factory=list)   # ← correct pattern

    def enroll(self, student_name: str) -> bool:
        if len(self.enrolled) < self.max_students:
            self.enrolled.append(student_name)
            return True
        return False

    @property
    def available_seats(self) -> int:
        return self.max_students - len(self.enrolled)


# ── Example 4: Dataclass with __post_init__ validation
class ExperienceLevel(str, Enum):
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"


@dataclass
class MLExperiment:
    """
    Validates data after __init__ using __post_init__.
    Java equivalent: validation logic inside the constructor body.
    """
    name: str
    algorithm: str
    accuracy: float
    level: ExperienceLevel = ExperienceLevel.BEGINNER
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 <= self.accuracy <= 1.0):
            raise ValueError(f"accuracy must be in [0, 1], got {self.accuracy}")
        # Auto-set level based on accuracy
        if self.accuracy > 0.90:
            object.__setattr__(self, "level", ExperienceLevel.ADVANCED)
        elif self.accuracy > 0.70:
            object.__setattr__(self, "level", ExperienceLevel.INTERMEDIATE)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Example 5: Nested dataclasses ────────────────────
@dataclass
class DataPoint:
    """A single labeled feature vector."""
    features: list[float]
    label: str
    weight: float = 1.0

    @property
    def dimension(self) -> int:
        return len(self.features)


@dataclass
class Dataset:
    """
    Collection of DataPoints — composition pattern.
    Java: class Dataset implements Iterable<DataPoint>
    """
    name: str
    points: list[DataPoint] = field(default_factory=list)

    def add(self, point: DataPoint) -> None:
        self.points.append(point)

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def labels(self) -> list[str]:
        return [p.label for p in self.points]

    def __iter__(self):
        return iter(self.points)

    def __len__(self) -> int:
        return self.size


# ══════════════════════════════════════════════════════
# PART 2 — PYDANTIC v2 MODELS
# Java equivalent: Bean Validation (@NotNull, @Size, @Valid)
# ══════════════════════════════════════════════════════

class TrainingRequest(BaseModel):
    """
    Validates incoming API request to train a model.
    Java: @Valid @RequestBody TrainingRequest in Spring MVC.

    Pydantic raises ValidationError (422 in FastAPI) if any field fails.
    """
    model_name: str   = Field(min_length=1, max_length=100)
    algorithm: str    = Field(default="logistic_regression")
    test_size: float  = Field(default=0.2, ge=0.1, le=0.5)
    features: list[str]
    target: str

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        allowed = {"logistic_regression", "random_forest", "svm", "linear_regression"}
        if v not in allowed:
            raise ValueError(f"Must be one of {sorted(allowed)}, got '{v}'")
        return v

    model_config = {"str_strip_whitespace": True}


class ModelMetrics(BaseModel):
    """Evaluation output from a trained model."""
    accuracy:  Optional[float] = Field(default=None, ge=0, le=1)
    precision: Optional[float] = Field(default=None, ge=0, le=1)
    recall:    Optional[float] = Field(default=None, ge=0, le=1)
    f1_score:  Optional[float] = Field(default=None, ge=0, le=1)
    auc_roc:   Optional[float] = Field(default=None, ge=0, le=1)
    mse:       Optional[float] = Field(default=None, ge=0)

    @property
    def is_classification(self) -> bool:
        return self.accuracy is not None


class PredictionResponse(BaseModel):
    """API response for a single prediction."""
    prediction:    str | float
    confidence:    float = Field(ge=0, le=1)
    model_name:    str
    features_used: list[str]
    timestamp:     datetime = Field(default_factory=datetime.now)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class DatasetStats(BaseModel):
    """Summary statistics about a dataset — validated on construction."""
    num_samples:        int = Field(gt=0)
    num_features:       int = Field(gt=0)
    num_classes:        Optional[int] = None
    class_distribution: dict[str, int] = Field(default_factory=dict)
    missing_values:     int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_distribution_totals(self) -> "DatasetStats":
        if self.class_distribution:
            total = sum(self.class_distribution.values())
            if total != self.num_samples:
                raise ValueError(
                    f"class_distribution sums to {total}, expected {self.num_samples}"
                )
        return self


class AppSettings(BaseSettings):
    """
    Reads config from environment variables.
    Java equivalent: @ConfigurationProperties (Spring Boot).

    Usage:
        export APP_MODEL_SERVER_PORT=9000
        settings = AppSettings()          # reads from env
    """
    model_server_host: str = "localhost"
    model_server_port: int = 8000
    max_batch_size:    int = 32
    log_level:         str = "INFO"
    debug_mode:        bool = False

    model_config = {"env_prefix": "APP_"}


# ══════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 60)
    print("DATACLASS DEMOS")
    print("=" * 60)

    # 1. Student
    s = Student("Alice", 25, "alice@example.com", gpa=3.5)
    print(f"\n1. {s}, passing={s.is_passing()}")

    # 2. Frozen Coordinate — mutation raises FrozenInstanceError
    c1, c2 = Coordinate(0.0, 0.0), Coordinate(3.0, 4.0)
    print(f"\n2. Distance: {c1.distance_to(c2):.2f}")
    try:
        c1.x = 99.0   # type: ignore
    except Exception as e:
        print(f"   Frozen! {type(e).__name__}: cannot mutate")

    # 3. Course with enrollment cap
    course = Course("CS101", "Intro to AI", max_students=2)
    print(f"\n3. Enroll Alice: {course.enroll('Alice')}")
    print(f"   Enroll Bob:   {course.enroll('Bob')}")
    print(f"   Enroll Carol: {course.enroll('Carol')}  ← full")
    print(f"   Seats left: {course.available_seats}")

    # 4. MLExperiment auto-level from __post_init__
    exp = MLExperiment("run_01", "RandomForest", accuracy=0.92)
    print(f"\n4. Experiment level auto-set: {exp.level.value}")
    try:
        MLExperiment("bad", "SVM", accuracy=1.5)
    except ValueError as e:
        print(f"   Caught bad accuracy: {e}")

    # 5. Dataset of DataPoints
    ds = Dataset("XOR")
    for feat, lbl in [([0, 0], "F"), ([1, 1], "F"), ([0, 1], "T"), ([1, 0], "T")]:
        ds.add(DataPoint(feat, lbl))
    print(f"\n5. Dataset size={ds.size}, labels={ds.labels}, dim={ds.points[0].dimension}")

    print("\n" + "=" * 60)
    print("PYDANTIC DEMOS")
    print("=" * 60)

    # Valid request
    req = TrainingRequest(
        model_name="iris_clf",
        algorithm="random_forest",
        features=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        target="species",
    )
    print(f"\nValid request → {req.model_name} | algo={req.algorithm}")
    print(f"JSON:\n{req.model_dump_json(indent=2)}")

    # Invalid algorithm
    try:
        TrainingRequest(model_name="x", algorithm="neural_net", features=["f1"], target="y")
    except Exception as e:
        print(f"\nValidation error (expected):\n{e}")

    # DatasetStats with cross-field validator
    stats = DatasetStats(
        num_samples=150, num_features=4, num_classes=3,
        class_distribution={"setosa": 50, "versicolor": 50, "virginica": 50},
    )
    print(f"\nDatasetStats OK: {stats.num_samples} samples, {stats.num_features} features")

    # Bad distribution (sum ≠ num_samples)
    try:
        DatasetStats(num_samples=100, num_features=4,
                     class_distribution={"A": 60, "B": 30})  # sums to 90
    except Exception as e:
        print(f"Cross-field error (expected): {e}")
