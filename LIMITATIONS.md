# SNN ADMM Optimizer: Known Limitations

This document outlines the current boundaries of the SNN ADMM Optimizer.

## ⚠️ Current Limitations

While the core ADMM optimization mechanics are fully functional, there are a few architectural and technical limitations in the current version:

**Authomatic use_reset**: Use reset False should be authomatically set to the last layer. Or somehow updates should be changed so this param is unnecessary. This also yields the question of having spiking nn that do not accumulate in the output layer (but this is a complete other world).

**Pooling standard linear**: All pooling methods flatten the output to a 2D shape for compatibility with linear layers by default. This must be addressed.

**Multibatch scheduling**: Allow multibatch scheduling.

**Make manager spiking agnostic**: At the moment the layer being spiking or not is known by two objects: layer and manager. This doubles down information and could possibly generate errors.

**Separate activation functions from layers**: At the moment activation functions are part of the layer. However I beleive that this should be separated for better modularity, and better initialization.

**Get rid of broadcasting functions**: Broadcasting functions like broadcast to match or format bias can lead to errors and make reading harder, even if they let us develop fast. I think having more clear shape handling in every step could be bug proof or at least this should be checked. 

**Loss function review**: Loss function module works but is a bit not intuitive and seems dirty to me. Could benefit from a rework.

**Better testing**: If we want to grow bigger and do it consistently without having bugs, we should improve or testing strategy, adding more tests that allow for early bug caching.

