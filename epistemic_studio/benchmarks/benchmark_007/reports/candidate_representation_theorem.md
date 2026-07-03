# Candidate Representation Theorem

For any system that accumulates reliable knowledge over time, there exists an abstract representation with:

- state S_t retaining information from prior interactions;
- environment interaction E_t producing observations/outcomes;
- update operator U(S_t, E_t) -> S_{t+1};
- evaluation relation R that is truth-correlated enough to distinguish improvement;
- scope relation C describing where retained changes transfer;
- stability constraint preventing U from systematically destroying previously valid retained structure.

The original five axioms specify S, E, R, C, and cost pressure, but not the truth-correlation or stability constraints strongly enough.
