# Event index requirements

- One local process writes at most 20 events per second.
- The dataset is expected to remain below 5 GB for the next year.
- Operators prefer a zero-service local installation.
- A later hosted edition may need concurrent writers and read replicas.
- The first release values recoverability and simple backups over horizontal scale.
