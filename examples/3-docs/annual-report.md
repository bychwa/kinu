# Acme 2025 Annual Report — Infrastructure

## Cloud Infrastructure

Cloud infrastructure spend for FY2025 totaled $4.2M, up 38% year over year. The
largest driver was object storage growth following the migration of customer
embeddings to S3 in Q2. Finance has approved a ceiling of $6.0M for FY2026,
with a review triggered at 80% consumption.

## Cost Per Tenant

Storage costs per tenant fell 22% after enabling Intelligent-Tiering and
shortening the lifecycle policy on intermediate embedding artifacts. The
median tenant now costs $0.31 per month to store and $0.04 per month to query.

## Headcount

Engineering grew from 41 to 58 people. Two thirds of the growth was in the
retrieval and data platform teams, reflecting the decision to bring search
infrastructure in-house rather than license a managed vector database.

## Outlook

The board has asked for a plan that keeps infrastructure cost growth below
revenue growth for the next four quarters. The primary lever is storage
layout: keeping hot indexes on fast media while pushing cold segments to
object storage rather than provisioning for peak.
