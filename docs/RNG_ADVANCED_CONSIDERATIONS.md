# RNG State Management: Advanced Considerations

This note pertains to the RNG state management implemented in `src/system_orchestrator.py`.

## Multi-Processing and Multi-Threading

If Project Crypto Bot Peter is ever adapted for multi-process or heavily multi-threaded execution, the current global RNG state restoration mechanism for Python's built-in `random` module and `numpy.random` (which relies on setting global state for these libraries) may require careful review and potential modification.

**Potential Issues in Concurrent Environments:**

*   **State Overlap/Interference:** Global RNG state, when restored or manipulated by one process/thread, could unintentionally affect others if they share the same Python interpreter and memory space without proper isolation.
*   **Predictability:** Ensuring predictable and reproducible random number sequences across independent processes or threads becomes more complex with global state.

**Possible Solutions/Approaches to Consider:**

*   **Per-Process/Per-Thread RNG Instances:** Each process or major thread could manage its own independent instances of `random.Random()` and `numpy.random.RandomState()`.
*   **Seeding Strategies:** Careful seeding strategies would be required for these independent instances to ensure they are either identically seeded (for exact replication of parallel tasks) or uniquely seeded (to ensure statistical independence).
*   **State Management:** Saving and loading RNG state would then need to handle a collection of states, possibly keyed by process/thread identifiers.

The current implementation is robust for single-process execution. These considerations are primarily for future architectural changes involving significant concurrency.
```
