# Benchmark Research #004 Final Report

## Which organizational principles survived historical comparison?

- applicability_boundaries: survived. Success mean 0.807, control mean 0.2, prevalence 1.0.
- persistent_memory: survived. Success mean 0.823, control mean 0.477, prevalence 1.0.
- institutionalized_criticism: survived. Success mean 0.77, control mean 0.227, prevalence 0.875.
- preserved_failures: survived. Success mean 0.676, control mean 0.27, prevalence 0.625.
- knowledge_compression: survived. Success mean 0.774, control mean 0.193, prevalence 0.75.
- question_refinement: survived. Success mean 0.774, control mean 0.27, prevalence 0.875.

## Which failed?

- role_separation: failed as universal. It appears useful in some systems but absent or weak in successful individual/theoretical cases. Counterexamples: ['darwin'].
- knowledge_graph_equivalent: failed as universal. It appears useful in some systems but absent or weak in successful individual/theoretical cases. Counterexamples: ['alphago'].

## Which remain uncertain?

- append_only_accumulation: uncertain. Score 0.715; counterexamples none coded, but expression varies strongly by domain.
- planning_prioritization: uncertain. Score 0.772; counterexamples none coded, but expression varies strongly by domain.

## If a civilization had to rebuild science from scratch, which principles first?

1. Persistent public memory of claims, methods, evidence, and revisions. Evidence: Darwin's notebooks, Linux mailing lists/version history, CERN data preservation, NASA lessons learned, and CRISPR's cumulative literature all show that durable memory lets later work reuse and correct earlier work.
2. Institutionalized adversarial criticism with preserved failures. Evidence: Linux patch review, CERN collaboration checks, Apollo lessons-learned practice, and the negative controls Theranos/Challenger/replication crisis show that weak criticism or hidden failure memory produces unreliable knowledge.
3. Compression through question refinement with explicit applicability boundaries. Evidence: Darwin compressed biological diversity into natural selection while working through species boundaries, CRISPR compressed bacterial immune machinery into guide-RNA genome editing with biochemical scope limits, AlphaGo compressed play into policy/value/search machinery inside the Go domain, CERN encodes likelihood/data applicability, and Bell Labs repeatedly produced reusable primitives with engineering contexts.

The Studio's three principles survive, but only after translation. Literal append-only software, journals, and graphs are not universal. Durable memory, adversarial correction, and compression under refined questions are much more general.

## Evidence base

- Memories: A Personal History of Bell Telephone Laboratories: https://quello.msu.edu/divi/wp-content/uploads/2015/08/Memories-Noll.pdf
- NASA Apollo Lunar Module Reliability Lessons Learned: https://llis.nasa.gov/lesson/1806
- NASA: Better Mechanisms Needed for Sharing Lessons Learned: https://www.gao.gov/products/gao-02-195
- Manhattan Project overview: https://ethos.lps.library.cmu.edu/article/id/35/
- Linux kernel submitting patches guide: https://www.kernel.org/doc/html/v4.16/process/submitting-patches.html
- Linux Kernel Mailing List archive: https://lkml.org/
- Mastering the game of Go with deep neural networks and tree search: https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/
- Nobel Prize in Chemistry 2020 press release: https://www.nobelprize.org/prizes/chemistry/2020/press-release/
- CRISPR Timeline: https://www.broadinstitute.org/what-broad/areas-focus/project-spotlight/crispr-timeline
- Darwin's species notebooks: https://www.darwinproject.ac.uk/commentary/evolution/darwin-s-species-notebooks-i-think
- Darwin's notebooks and reading lists: https://darwin-online.org.uk/EditorialIntroductions/vanWyhe_notebooks.html
- CERN Open Data: https://openscience.cern/open-data
- Open science at CERN: https://cerncourier.com/a/open-science-a-vision-for-collaborative-reproducible-and-reusable-research/
- Lessons from Theranos: https://pmc.ncbi.nlm.nih.gov/articles/PMC8979578/
- Theranos and peer review case study: https://www.bsm.upf.edu/documents/2024-case-study-elisabeth-holmes-theranos.pdf
- Challenger disaster and normalization of deviance: https://magazine.columbia.edu/article/challenger-disaster-normalization-deviance
- Replication crisis in psychology: https://nobaproject.com/modules/the-replication-crisis-in-psychology
