# Configuration Model

## Summary
The backend loads configuration from hardcoded defaults, a required `config.yaml`, and explicit environment overrides. Secrets do not belong in committed files.

Git policy:
- `config.yaml.example`: committed
- `config.yaml`: ignored
- `.env.example`: committed
- `.env`: ignored

Recommended ignore rules:

```gitignore
/.env
/.env.*
!/.env.example

/config.yaml
/config.*.yaml
!/config.yaml.example
```

## Dependencies
- `pyyaml`
- `python-dotenv`
- `pydantic-settings`

Rules:
- use safe PyYAML loading semantics
- use `load_dotenv(override=False)`
- do not implement a custom dotenv parser

## File Selection
The active configuration file is chosen by:

1. `CONFIG_FILE` from the process environment
2. otherwise `./config.yaml`

`CONFIG_FILE` is not read from YAML because YAML has not been located yet.

## Exact Loading Algorithm
Startup follows this order:

1. load hardcoded defaults
2. determine the active config file path
3. parse `config.yaml`
4. load `.env` into the process environment with `load_dotenv(override=False)`
5. resolve exact `${ENV_VAR_NAME}` placeholders inside YAML string values
6. build explicit environment-variable overrides
7. deep-merge:
   - defaults
   - resolved YAML
   - explicit environment overrides
8. validate the merged object with the root Pydantic settings model
9. convert secret-bearing values to `SecretStr`
10. fail startup if parsing, substitution, merge, or validation fails

Precedence is fixed:
- shell or deployment environment > `.env`
- explicit environment overrides > resolved YAML > hardcoded defaults

## Placeholder Contract
Placeholder resolution and environment overrides are separate mechanisms.

### Required placeholders
- exact full-string placeholders only, such as `${MODEL_API_KEY}`
- `${VAR}` means the variable is required
- missing required values fail startup
- empty required values fail startup
- no unresolved `${...}` text may survive startup
- partial forms like `prefix-${VAR}` are rejected
- unsupported forms like `$VAR` are rejected

### Optional values
Optional values should use:
- `null`
- omission
- literal defaults

Example:

```yaml
openrouter:
  site_url: null
  app_name: PSVIEW Recruiting Agent
```

Required secret-backed values use placeholders:

```yaml
model:
  api_key: ${MODEL_API_KEY}
  model_name: ${MODEL_NAME}
```

## Environment Override Contract
Supported flat environment aliases:

- `CONFIG_FILE`
- `APP_NAME`
- `APP_ENV`
- `APP_VERSION`
- `API_V1_PREFIX`
- `LOG_LEVEL`
- `MODEL_PROVIDER`
- `MODEL_API_KEY`
- `MODEL_BASE_URL`
- `MODEL_NAME`
- `MODEL_GENERAL_CHAT_NAME`
- `MODEL_STRUCTURED_JSON_NAME`
- `MODEL_CODING_BACKEND_NAME`
- `MODEL_STRUCTURED_OUTPUT_MODE`
- `MODEL_TIMEOUT_SECONDS`
- `MODEL_MAX_RETRIES`
- `MODEL_MAX_OUTPUT_TOKENS`
- `MODEL_TEMPERATURE`
- `MODEL_REPAIR_ATTEMPTS`
- `MODEL_CONCURRENCY_LIMIT`
- `OPENROUTER_SITE_URL`
- `OPENROUTER_APP_NAME`
- `ALLOWED_ORIGINS`
- `MAX_REQUEST_BODY_BYTES`
- `MAX_HISTORY_MESSAGES`
- `MAX_CONVERSATION_TURNS`
- `MAX_RESPONSE_CHARACTERS`
- `MAX_REVISION_ATTEMPTS`
- `LANGGRAPH_RECURSION_LIMIT`
- `RETRIEVAL_ENABLED`
- `RETRIEVAL_TOP_K`
- `RETRIEVAL_MIN_SCORE`
- `RETRIEVAL_REUSE_PENALTY`
- `RETRIEVAL_MAX_FACT_CANDIDATES`

Nested mapping:
- `APP_*` -> `app.*`
- `MODEL_*` -> `model.*`
- `OPENROUTER_*` -> `openrouter.*`
- runtime limit aliases -> `runtime.*`
- retrieval aliases -> `retrieval.*`

List parsing:

```env
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

becomes:

```python
["http://localhost:5173", "http://localhost:3000"]
```

Empty optional environment variables are ignored rather than used to erase valid YAML values.

## Workload-Specific Model Routing
The backend may route different workloads to different configured models while keeping one
provider-compatible gateway.

Supported routing fields:
- `model.model_name`: fallback model used when no workload-specific model is set
- `model.general_chat_model_name`: candidate-facing response generation and revision
- `model.structured_json_model_name`: structured extraction, planning, and evaluation
- `model.coding_backend_model_name`: reserved for future technical or code-oriented workloads

Supported environment overrides:
- `MODEL_NAME`
- `MODEL_GENERAL_CHAT_NAME`
- `MODEL_STRUCTURED_JSON_NAME`
- `MODEL_CODING_BACKEND_NAME`

Current routing policy:
- company configuration uses the structured-JSON model
- outreach planning uses the structured-JSON model
- candidate-reply analysis uses the structured-JSON model
- action planning uses the structured-JSON model
- response evaluation uses the structured-JSON model
- candidate-facing response generation uses the general-chat model
- candidate-facing response revision uses the general-chat model
- coding/backend routing is available for future technical flows and falls back to `MODEL_NAME`

## Example Files
### `config.yaml.example`
```yaml
app:
  name: PSVIEW Recruiting Agent API
  env: development
  version: 0.1.0
  api_v1_prefix: /api/v1
  log_level: INFO

model:
  provider: openrouter
  api_key: ${MODEL_API_KEY}
  base_url: https://openrouter.ai/api/v1
  model_name: ${MODEL_NAME}
  general_chat_model_name: null
  structured_json_model_name: null
  coding_backend_model_name: null
  structured_output_mode: auto
  timeout_seconds: 45
  max_retries: 2
  max_output_tokens: 1800
  temperature: 0.2
  repair_attempts: 1
  concurrency_limit: 4
  extra_body: {}

openrouter:
  site_url: null
  app_name: PSVIEW Recruiting Agent

runtime:
  allowed_origins:
    - http://localhost:5173
  max_request_body_bytes: 100000
  max_history_messages: 20
  max_conversation_turns: 16
  max_response_characters: 1000
  max_revision_attempts: 1
  langgraph_recursion_limit: 20

retrieval:
  enabled: true
  top_k: 5
  min_score: 0.05
  reuse_penalty: 0.15
  max_fact_candidates: 20
```

### `.env.example`
```env
CONFIG_FILE=config.yaml

MODEL_API_KEY=
MODEL_NAME=

MODEL_PROVIDER=
MODEL_BASE_URL=
MODEL_GENERAL_CHAT_NAME=
MODEL_STRUCTURED_JSON_NAME=
MODEL_CODING_BACKEND_NAME=
MODEL_STRUCTURED_OUTPUT_MODE=
MODEL_TIMEOUT_SECONDS=
MODEL_MAX_RETRIES=
MODEL_MAX_OUTPUT_TOKENS=
MODEL_TEMPERATURE=
MODEL_REPAIR_ATTEMPTS=
MODEL_CONCURRENCY_LIMIT=

OPENROUTER_SITE_URL=
OPENROUTER_APP_NAME=

APP_NAME=
APP_ENV=
APP_VERSION=
API_V1_PREFIX=
LOG_LEVEL=

ALLOWED_ORIGINS=
MAX_REQUEST_BODY_BYTES=
MAX_HISTORY_MESSAGES=
MAX_CONVERSATION_TURNS=
MAX_RESPONSE_CHARACTERS=
MAX_REVISION_ATTEMPTS=
LANGGRAPH_RECURSION_LIMIT=

RETRIEVAL_ENABLED=
RETRIEVAL_TOP_K=
RETRIEVAL_MIN_SCORE=
RETRIEVAL_REUSE_PENALTY=
RETRIEVAL_MAX_FACT_CANDIDATES=
```

## Configuration Modules
- `src/psview_agent/core/config.py`
- `src/psview_agent/core/config_loader.py`
- `src/psview_agent/core/config_merge.py`
- `src/psview_agent/core/env_placeholders.py`
- `src/psview_agent/core/config_redaction.py`

Responsibilities:
- `config.py`: typed settings, enums, validators, cached accessor
- `config_loader.py`: locate config file, parse YAML, load dotenv, merge, validate
- `config_merge.py`: deep merge and flat env alias mapping
- `env_placeholders.py`: exact placeholder resolution
- `config_redaction.py`: safe diagnostics without leaking secrets

## Required Test Cases
- shell environment wins over `.env`
- `.env` does not overwrite existing process environment values
- empty optional environment variables do not overwrite YAML
- empty required placeholder values fail startup
- placeholders resolve in nested mappings and lists
- partial `prefix-${VAR}` and unsupported `$VAR` syntax are rejected
- YAML root must be a mapping
- duplicate keys fail clearly
- secret values do not appear in repr, diagnostics, or startup errors
- production wildcard CORS fails
- provider-specific validation behaves correctly
- custom `CONFIG_FILE` paths work in tests and Docker
