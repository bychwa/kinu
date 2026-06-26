# S3 Cost Optimization Notes

## Storage Classes

S3 Standard is the default and the most expensive per gigabyte. Standard-IA
costs roughly half as much for storage but adds a retrieval fee and a 30-day
minimum duration. Glacier Instant Retrieval is cheaper still and still returns
objects in milliseconds, which makes it a reasonable home for indexes that are
queried rarely but must stay fast.

## Intelligent-Tiering

Intelligent-Tiering moves objects between access tiers automatically. For
workloads with unpredictable access patterns this typically reduces storage
cost by 20-40% versus Standard, at the cost of a small per-object monitoring
fee. The monitoring fee makes it a poor fit for namespaces holding millions of
tiny objects; batch objects before enabling it.

## Lifecycle Policies

Lifecycle rules expire intermediate artifacts automatically. Embedding
artifacts, temporary exports, and superseded index generations should all have
an expiry measured in days, not months. A single missing rule is the most
common cause of a surprise storage bill.

## Request Costs

Requests are billed separately from storage and are frequently the larger line
item for small-object workloads. Reading 10,000 tiny objects costs far more
than reading one large object of the same total size. Batch writes and read
wide contiguous ranges rather than many small ones.
