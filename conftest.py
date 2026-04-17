"""
conftest.py — pytest configuration for the infini-gram test suite.

Registers the --builder flag used by test_infini_gram.py to swap between
suffix-array implementations:

    pytest test_infini_gram.py -v --builder=sorted
    pytest test_infini_gram.py -v --builder=sais      (default)
    pytest test_infini_gram.py -v --builder=rust
    pytest test_infini_gram.py -v --builder=caps_sa
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--builder',
        action='store',
        default='sais',
        choices=['sorted', 'sais', 'rust', 'caps_sa'],
        help='SA builder: sorted | sais | rust | caps_sa  (default: sais)',
    )


@pytest.fixture(scope='session')
def builder(request):
    return request.config.getoption('--builder')
