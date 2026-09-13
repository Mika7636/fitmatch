"""Session 3: the HTTP layer over the session-2 recommendation cascade.

The API adds no recommendation logic.  It loads
:class:`~src.recommender.pipeline.FitMatchRecommender` once at startup,
translates JSON into the profile dict that
:mod:`src.recommender.profiles` already knows how to normalise, and
translates the cascade's own report objects back into JSON.  If a number in a
response is not in :mod:`src.recommender`, it is not a number the API made up.

Two session-2 findings shape the response shape, and both are in the contract
rather than in a debug field:

* **Relaxation is the normal path.**  The median user's feasible set with
  every soft constraint applied is 0, and colour is relaxed for 98.7% of
  users.  So every ``/api/recommend`` response carries the full relaxation
  log, and an empty ``recommendations`` list is always accompanied by the
  account of what was tried.
* **85.2% of users get the cold-start ranker.**  So every response states
  which of the two content modes produced the ordering.
"""

from src.api.main import app, create_app

__all__ = ["app", "create_app"]
