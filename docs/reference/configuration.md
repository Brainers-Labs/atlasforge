# Configuration file

An optional `atlasforge.toml` in your project sets the defaults every run in it shares: which
backend, which endpoint, which model. Without one, nothing changes — every setting has a built-in
default and every command works from flags alone.

## The file

```toml
[atlasforge]
backend = "openai"
base_url = "http://127.0.0.1:8000/v1"
model = "NCAIR1/N-ATLaS"
```

Only the `[atlasforge]` table is read, so the file may hold other tools' settings too.

{{ config_table() }}

## Where it is looked for

The file is read from the **nearest directory at or above the one you run in**. A project root
holds it once and commands run from any subdirectory pick it up.

Nothing is read from your home directory or anywhere system-wide. A run made in a repository is
reproducible from that repository alone, and a file you did not write cannot change your settings.

## Precedence

Highest wins:

1. the flag you type (`--model ...`)
2. the environment variable, where the option has one (`ATLASFORGE_BASE_URL`)
3. `atlasforge.toml`
4. the built-in default

So a file never overrides something you asked for on the command line:

```bash
atlasforge run "Kwana biyu" --model my-fine-tune   # the flag wins
```

## What is deliberately not in the file

- **`allow_insecure_http`.** Permitting plain http to a non-local server should be typed on the
  command line and visible in your shell history, not a property of a checked-in file.
- **Your API key.** Use `ATLASFORGE_API_KEY` or `HF_TOKEN` from the environment
  ([environment variables](environment.md)), so a token never lands in a file you might commit.
- **Anything not in the table above.** An unknown key is an error naming the valid ones, not a
  silent no-op — a setting that quietly does nothing is worse than one that stops the run.

## Checking what is in effect

`atlasforge doctor` has a `config` row that names the file it found and the settings it will
apply, so a stale or unexpected file is visible rather than mysterious. With no file the row says
so; with a broken one the check fails and shows the reason and the line to fix.
