#!/bin/bash

curl -s \
    "https://api.groq.com/openai/v1/audio/transcriptions"   \
      -H "Authorization: Bearer ${GROQ_API_KEY}"            \
      -F "model=whisper-large-v3"                           \
      -F "file=@${1}"                                       \
      -F "temperature=0"                                    \
      -F "response_format=verbose_json"                     \
      -X POST | jq .text
