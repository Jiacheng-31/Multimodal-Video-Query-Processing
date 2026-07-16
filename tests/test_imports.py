def test_public_modules_import() -> None:
    import clipplan.api.scorer  # noqa: F401
    import clipplan.data.prepare  # noqa: F401
    import clipplan.evaluation.evaluate  # noqa: F401
    import clipplan.retrieval.retrieve_candidates  # noqa: F401
    import clipplan.router.env  # noqa: F401
