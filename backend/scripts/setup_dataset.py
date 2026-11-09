"""Prevent the old proxy generator from overwriting real NIH data."""


def create_proxy_dataset(*args, **kwargs):
    raise RuntimeError(
        "Proxy data generation into the real dataset is disabled. Use temporary artificial fixtures in tests only."
    )


if __name__ == "__main__":
    create_proxy_dataset()
