"""Allow `python -m morseformer.cli` invocation."""

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
