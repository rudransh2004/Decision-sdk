# Decision SDK

Decision SDK is a Python library for typed, runtime-defined decisions. You
give a model a state, a question, and the options you want it to choose
between, and it returns probabilities over those options instead of free-form
text. The same `choice`, `noul` (yes/no), and `score` calls work on local
models, vLLM, and a remote server.

- [Getting started](getting-started.md): install, make a decision, call a server.
- [Run decisions](inference.md): results, abstention, and batches.
- [Serve decisions over HTTP](serving.md): the server, the clients, and errors.
- [Datasets and training](datasets-and-training.md): data format, SFT, calibration, RLCD.
- [Convert an LLM](conversion.md): turn a plain causal LM into a decision model.
- [Model support](model-support.md): per-family notes and the noul order.
- [vLLM and inference backends](inference-backends.md): which models run under vLLM.
- [Benchmarks and run logs](benchmarks.md): what was run and what came out.
- [API reference](api.md): every public class and function.

The project README has the support matrix, and the
[CHANGELOG](https://github.com/rudransh2004/Decision-sdk/blob/main/CHANGELOG.md)
lists what changed in each release.
