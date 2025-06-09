## AutoFL

#### Phase 1 -> Simple Federated Continual Learning 

- [x]  Implement FL with Flower
- [x]  Understand how CL can work on each client
- [x]  Setup Individual client CL with Avalanche
- [x]  Integrate Avalanche with Flower
- [ ]  Basic Tests to ensure successful integration
- [ ]  Extend Base for Benchmarks
    - [x]  Add Workloads → Look at [CIFAR10CL.py] on dev --> **implemented with this branch**
    - [ ]  Add Models → MobileNet, ResNet etc
    - [ ]  Add FL Strats
    - [ ]  Add CL Strats → Buffer Replay, EWC, Hybrid
    - [ ]  Add provisions for NIID
- [ ]  Make robust evaluation metrics and logging
    - [x]  Currently we are just printing and logging cl metrics of each experience and stream and then the final aggregate metrics for FL
    - [x]  Understanding which FL and CL metrics are required
    - [x]  Integrating and Logging using WandB --> **fixes done**
    - [ ]  Provision for local backup → not really  necessary immediately
- [x]  Centralize Configuration
    - [x]  Temp through yaml file
    - [ ]  Long term → Integrate Hydra
- [ ]  Adding documentation
    - [x] Doccumentation for DL and workloads 
    - [ ] Working FCL vs CL etc