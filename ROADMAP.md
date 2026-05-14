# SNN ADMM Optimizer: Roadmap & Known Limitations

This document outlines the current boundaries of the SNN ADMM Optimizer and our planned milestones for future releases. 

## ⚠️ Current Limitations

While the core ADMM optimization mechanics are fully functional, there are a few architectural and technical limitations in the current version:

**Make manager spiking agnostic**: At the moment the layer being spiking or not is known by two objects: layer and manager. This doubles down information and could possibly generate errors.

**Separate activation functions from layers**: At the moment activation functions are part of the layer. However I beleive that this should be separated for better modularity, and better initialization.

**Check z_to_use**: Check if this functionality makes sense in production.

## 🚀 Future Work

Our goal is to make the ADMM Optimizer the standard for gradient-free training in neural networks. Here are the features and improvements planned for upcoming releases:
