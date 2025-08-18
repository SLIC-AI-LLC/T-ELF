from collections import defaultdict
from typing import Any, Dict, Sequence, Optional, Iterator, Tuple

SAVE_DIR_BUNDLE_KEY = 'save_path'
SOURCE_DIR_BUNDLE_KEY = 'dir'
DIR_LIST_BUNDLE_KEY = 'directories'
RESULTS_DEFAULT = 'results'

class DataBundle:
    """
    Stores values under base_keys, each with tagged versions and a '_latest' pointer.
    Now supports:
      - bundle.tag            → NamespaceView for that tag
      - bundle['tag']         → NamespaceView for that tag
      - bundle.tag.key        → same as bundle['tag']['key']
    """
    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        super().__setattr__('_store', defaultdict(dict))
        if initial:
            for k, v in initial.items():
                if "." in k:
                    tag, base = k.split(".", 1)
                else:
                    tag, base = "Init", k
                self.__setitem__(base, v, tag=tag)

    def tags(self) -> set[str]:
        """Return all tags currently in the bundle."""
        tags = set()
        for bucket in self._store.values():
            tags.update(t for t in bucket.keys() if t != '_latest')
        return tags

    def keys_by_tag(self, tag: str) -> list[str]:
        """Return all base-keys that have a value under `tag`."""
        return [base for base, bucket in self._store.items() if tag in bucket]
    
    def print_tags_and_keys(self) -> None:
        """
        Pretty-print each tag and the base-keys under it.
        """
        for tag in sorted(self.tags()):
            keys = self.keys_by_tag(tag)
            print(f"{tag!r}: {keys}")

    def __getitem__(self, key: str) -> Any:
        # If they ask for a bare tag, return the NamespaceView
        if "." not in key and key in self.tags():
            return NamespaceView(self, key)

        # Otherwise fall back to normal behavior
        tag, base = self._split(key)
        bucket = self._store.get(base, {})
        if tag is None:
            tag = bucket.get('_latest')
            if tag is None:
                raise KeyError(f"No value for key {base!r}")
        if tag not in bucket:
            raise KeyError(f"No {base!r} produced by tag {tag!r}")
        return bucket[tag]

    def __setitem__(self, key: str, value: Any, *, tag: Optional[str] = None) -> None:
        assumed = tag or self.__class__.__name__
        tag, base = self._split(key, assume_tag=assumed)
        bucket = self._store.setdefault(base, {})
        bucket[tag] = value
        bucket['_latest'] = tag

    def __getattr__(self, name: str) -> Any:
        # If they access a tag by dot, return NamespaceView
        if name in self.tags():
            return NamespaceView(self, name)
        # If they access a base key by dot, return its latest value
        if name in self._store:
            return self.get(name)
        raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name!r}")

    def get(self, key: str, default: Any = None) -> Any:
        tag, base = self._split(key)
        bucket = self._store.get(base)
        if not bucket:
            return default
        if tag is None:
            tag = bucket.get('_latest')
            if tag is None:
                return default
        return bucket.get(tag, default)

    def as_dict(self) -> Dict[str, Any]:
        return {base: bucket[bucket['_latest']] for base, bucket in self._store.items()}

    def namespaced(self, tag: str) -> "NamespaceView":
        return NamespaceView(self, tag)

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except KeyError:
            return False

    def keys(self) -> Sequence[str]:
        return list(self._store.keys())

    def items(self) -> Iterator[Tuple[str, Any]]:
        for k in self._store:
            yield k, self.get(k)

    def values(self) -> Iterator[Any]:
        for k in self._store:
            yield self.get(k)

    def __iter__(self):
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self):
        latest = {base: bucket.get('_latest') for base, bucket in self._store.items()}
        return f"DataBundle(latest={latest})"

    def _split(self, key: str, assume_tag: Optional[str] = None) -> Tuple[Optional[str], str]:
        if "." in key:
            tag, base = key.split(".", 1)
        else:
            tag, base = None, key
        return tag or assume_tag, base

    def __delitem__(self, key: str) -> None:
        tag, base = self._split(key)
        if base not in self._store:
            raise KeyError(f"No such key: {key!r}")
        bucket = self._store[base]
        if tag is None:
            tag = bucket.get("_latest")
            if tag is None:
                raise KeyError(f"No latest tag for base key: {base!r}")
        if tag not in bucket:
            raise KeyError(f"Tag {tag!r} not found for base key {base!r}")
        del bucket[tag]
        if len(bucket) <= 1:
            del self._store[base]
        else:
            if bucket.get("_latest") == tag:
                remaining = [k for k in bucket if k != "_latest"]
                bucket["_latest"] = remaining[-1] if remaining else None

    def pop(self, key: str, default: Any = None) -> Any:
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if default is not None:
                return default
            raise

class NamespaceView:
    """
    Read-only view for a single tag.
      • view['foo'] → bundle['Tag.foo']
      • view.foo    → bundle['Tag.foo']
    """
    def __init__(self, bundle: DataBundle, tag: str):
        self._bundle = bundle
        self._tag = tag

    def __getitem__(self, base: str):
        return self._bundle[f"{self._tag}.{base}"]

    def __getattr__(self, name: str):
        # allow dot access
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"{self.__class__.__name__!r} has no key {name!r}") from e

    def __contains__(self, base: str):
        try:
            _ = self._bundle[f"{self._tag}.{base}"]
            return True
        except KeyError:
            return False

    def __repr__(self):
        keys = self._bundle.keys_by_tag(self._tag)
        return f"NamespaceView(tag={self._tag!r}, keys={keys})"
