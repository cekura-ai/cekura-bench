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

## Known limits

A provider that reports no usage gets no cost, and the cell says so rather than
showing zero. One provider loses its final usage report when the connection
closes before the turn completes, so its totals run low; that is disclosed on
the row rather than corrected by estimate, since we cannot measure what was
never sent.

Audio tokens cost several times text tokens, and cached input a fraction of
fresh input, so a price table that collapses them is not a price table. Every
rate here is per million tokens, matching how the vendors publish them.
