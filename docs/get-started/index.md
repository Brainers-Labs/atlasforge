# Get started

Pick the path that matches what you have today.

```mermaid
flowchart TD
    A[What do you have?] --> B{A model to test?}
    B -- "Not yet" --> C[Quickstart: run the offline demo]
    B -- "An OpenAI-compatible server" --> D[Evaluate a model]
    B -- "A GPU and the weights" --> E[Serve or load N-ATLaS]
    C --> F[Install, then connect a model]
    E --> D
    F --> D
    D --> G[Compare two models]
    G --> H[Fine-tune and publish]
```

| If you want to... | Go to |
|---|---|
| See what AtlasForge does, right now, with nothing to set up | [Quickstart](quickstart.md) |
| Install it, with or without the heavy extras | [Installation](installation.md) |
| Download N-ATLaS and understand the licence | [Access and licences](access-and-licences.md) |
| Decide how to run the models on your hardware | [Choose your setup](choose-your-setup.md) |

!!! tip "Never used it before?"
    Do the [Quickstart](quickstart.md) first. It takes about two minutes and every later page makes more sense afterwards.
