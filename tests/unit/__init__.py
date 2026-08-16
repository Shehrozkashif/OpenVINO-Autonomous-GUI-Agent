# Makes tests/unit a package so its module names cannot collide with
# tests/live (both hold a test_grounding.py). Without it, "pytest tests/"
# aborts collection with a duplicate-basename error.
