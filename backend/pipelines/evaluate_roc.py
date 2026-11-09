"""Legacy command forwards to strict held-out evaluation; no leaked NIH metrics."""

from backend.pipelines.eval import main

if __name__ == "__main__":
    main()
