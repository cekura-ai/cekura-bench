# Prices

A run record says what a call consumed. This says what that cost. They are kept
apart on purpose: consumption is a measurement taken during the call and can
never be recovered afterwards, while a price is a judgement about a vendor's
page on a date, differs by account and contract, and will be wrong eventually.
Splitting them means a published price can be corrected — or disputed by the
vendor — without re-running a single call.

`prices.json` carries one entry per row of the board. Every entry names the date
it was read and the page it was read from, because an undated price is not
reproducible and a benchmark that publishes one is asking to be believed rather
than checked.

## What is and is not included

Only the speech path: the model, and for a cascade its transcriber and its
voice. Not the transport, not the container the reference agent runs in, not the
platform that placed the call. Those are ours and are the same for every row, so
including them would move every number by the same amount and change no ranking,
while making the figure impossible for anyone else to reproduce.

One row carries a second model behind the first, and both are priced, because a
single number there would flatter it against rows that do all the work in one.

## How a call is priced

Every token count is split into the lanes vendors price apart: text and audio,
fresh and cached, input and output. The record keeps the framework's totals, in
which the audio and cached counts sit inside the prompt and output figures, so
each lane is a difference, and a lane below zero means the record does not mean
what the arithmetic assumes; that call is refused rather than priced. Reasoning
is inside the output total on some vendors and beside it on others, and each row
says which.

A row can carry more than one component: a model billed by the minute plus a
second model behind it billed by the token is priced as both, and the figure
shows each. A per-minute component names what it is measured on, because the
seconds a vendor reports are not the seconds a pipeline ran.

## Known limits

- A lane with tokens in it and no rate in the table is a refusal, never free.
- A record kept before a row's speech tokens were kept apart from its text
  cannot be priced: the total alone, read at the text rate, would look cheap.
- Some rows bill something the record cannot see, such as a transcription
  charge whose usage the framework drops. The row lists it under `unmetered`,
  and the figure travels with that note rather than with an estimate.
- A rate read from anywhere but the vendor's own page stays `verified: false`,
  and an unverified row is computed but not published.

Audio tokens cost several times text tokens, and cached input a fraction of
fresh input, so a price table that collapses them is not a price table. Every
token rate here is per million tokens, matching how the vendors publish them.
