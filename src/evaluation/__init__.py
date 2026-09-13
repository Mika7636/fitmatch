"""Session 5: offline evaluation of the session-2 cascade.

Nothing in this package is part of the recommender.  It imports
:mod:`src.recommender`, never the other way round, and it changes no scoring
behaviour -- it measures what sessions 1-4 already built and writes the
numbers down.

Two entry points::

    python -m src.evaluation.evaluate     five configurations, K = 5/10/20
    python -m src.evaluation.cold_start   the same metrics per rating-count segment

Read ``reports/circularity_note.md`` before reading any number either of them
prints.  The interaction log these metrics score against was synthesised by
``src/data/build_interactions.py`` from a fit function that shares all ten
Chapter 7 constraint fields and five of eight Chapter 3 user fields with the
recommender.  Offline accuracy here measures agreement between two hand-written
rule sets, not real-world recommendation quality.
"""
