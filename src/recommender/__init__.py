"""FitMatch recommendation engine (session 2).

Two techniques from the course's fixed list, built as separate, independently
testable modules and combined in a cascade:

* :mod:`src.recommender.knowledge_based` -- Knowledge-Based Constraint-Based
  Recommendation (Chapter 7).  Hard/soft constraint filtering with a defined
  relaxation order.  Decides *which* products are admissible.
* :mod:`src.recommender.content_based` -- Content-Based Filtering with
  TF-IDF + Cosine Similarity (Chapter 3).  Decides *how* the admissible
  products are ordered.
* :mod:`src.recommender.pipeline` -- the cascade that runs one then the other.

Neither module imports the other.  The pipeline is the only place they meet,
so each can be unit-tested on its own.
"""

from __future__ import annotations

__all__ = ["knowledge_based", "content_based", "pipeline", "profiles"]
