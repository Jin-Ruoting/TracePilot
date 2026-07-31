# BioML compatibility example

The original EpiCETRA experiment-planning workflow is retained as an optional
TracePilot task pack. It is not the default product workflow or the primary
evaluation target.

The example metrics and run records are deterministic workflow fixtures. They
must not be interpreted as published EpiCETRA results or independent scientific
validation.

The legacy example inputs remain at:

- `examples/epicetra_context.json`
- `examples/epicetra_experiment_log.json`

Use `tracepilot.taskpacks.build_legacy_bioml_agent()` when reproducing the
pre-0.2 deterministic BioML workflow.
