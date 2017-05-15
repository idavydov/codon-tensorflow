A simple implementation of codon models in tensorflow.
I don't think there's a good reason to use it seriously.

Although,
- the code is very simple compared to other implementations
- rather fast, since it uses a lot of matrix operations
- should translate well into GPU computations (cuda), tensorflow does this transparently
- it's uses reverse accumulation to compute gradients, automatically

on large trees this is much faster than fastcodeml, probably due to automatic differentiation.

The downsides are:
- There's an overhead in creating a computational graph. It is a problem especially for small datasets.
- I'm not sure if computed values are efficiently reused.
- Some of gradient elements are `nan`, I didn't debug this yet.

Two models are implemented: M0 and Branch-Site.
