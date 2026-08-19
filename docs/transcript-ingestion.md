# Transcript ingestion contract

The target agent must publish its completed provider transcript to Cekura and
associate it with the correct Cekura `run_id`. A test that receives no provider
call or transcript is a benchmark failure; it must not be silently dropped from
metrics.

At minimum, preserve the ordered spoken turns and native tool activity:

```json
[
  { "role": "agent", "content": "How can I help today?" },
  { "role": "user", "content": "I'd like to book an appointment." },
  {
    "role": "tool_call_invocation",
    "name": "lookup_patient",
    "arguments": "{\"phone\":\"2025550188\"}",
    "tool_call_id": "call_123"
  },
  {
    "role": "tool_call_result",
    "content": "{\"patient_id\":\"p_1002\"}",
    "tool_call_id": "call_123",
    "successful": true
  }
]
```

Use lowercase `function_call` / `function_call_result` if that is the format
expected by your selected Cekura custom-provider endpoint. Preserve the original
tool name, the raw argument object, ordering, and the call ID that links each
result to its invocation. Do not fabricate tool rows from speech.

For number-associated voice runs, persist the Cekura caller-number → `run_id`
association before hangup. Post-call text matching is not a reliable association
mechanism.

