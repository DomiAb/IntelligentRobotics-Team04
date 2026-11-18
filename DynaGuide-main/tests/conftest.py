def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the dynamics model, about a minute on the CPU")
