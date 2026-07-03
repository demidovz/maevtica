# Suggested Studio v0.2 Architecture

Build around mechanisms, not current modules:

- Persistent Research State
- Explorer separated from Engine
- Planner-guided prioritization
- Knowledge Graph
- Append-only memory
- Question refinement
- Compression
- Counterexample preservation

Drop or demote mechanisms that did not survive minimal search. Keep their implementation only when needed as a cheap support for a retained principle.
