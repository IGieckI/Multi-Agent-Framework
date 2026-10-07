"""
Download the Little Alchemy 2 recipe file from minqi/wordcraft and validate it
"""

import json
import urllib.request
from pathlib import Path
from framework.tasks.wordcraft.recipes import RecipeBook

COMMIT = "4d6a1b95b7b5690f45526220fd43cbf6b84e6ba1"
URL = f"https://raw.githubusercontent.com/minqi/wordcraft/{COMMIT}/datasets/alchemy2.json"
PATH = Path(__file__).resolve().parents[1] / "data" / "alchemy2.json"


def main() -> None:
    raw = urllib.request.urlopen(URL).read()
    PATH.parent.mkdir(exist_ok=True)
    PATH.write_bytes(raw)

    book = RecipeBook.load(PATH)
    entities = json.loads(raw)["entities"]
    listed = sum(len(item["recipes"]) for item in entities.values())
    print(f"saved {PATH}")
    print(f"items: {len(book.items)}")
    print(f"recipes listed: {listed}, distinct pairs: {len(book.pairs)}")
    print(f"recipes dropped (result is also an ingredient): {book.dropped}")
    print(f"pairs with more than one result: {sum(len(r) > 1 for r in book.pairs.values())}")
    unknown = {i for pair in book.pairs for i in pair} - set(book.items)
    if unknown:
        raise SystemExit(f"ingredients that are not items: {sorted(unknown)}")

if __name__ == "__main__":
    main()
