from atlas_http import Client


def fetch_order(order_id: str) -> dict:
    client = Client(timeout=5)
    return client.get(f"/orders/{order_id}")
