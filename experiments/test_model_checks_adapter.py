from adapters.model_checks import classification


def test_completed_negative_remains_scientific_evidence():
    result = classification(dict(status='FAIL', native_calls=351), 1)
    assert result['solver_numerical_pass'] is False
    assert result['execution_error'] is None
    assert result['outcome'] == 'checks_completed'


def test_partial_or_setup_failure_is_not_completed():
    for summary in (dict(status='FAIL', native_calls=34), dict(status='SETUP_FAIL'), {}):
        assert classification(summary, 1)['execution_error']
        assert classification(summary, 1)['solver_numerical_pass'] is None
