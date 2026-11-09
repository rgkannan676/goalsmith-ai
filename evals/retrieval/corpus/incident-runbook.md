# Production Incident Runbook

How the platform team handles outages of the Beacon cloud.

## Severity levels

SEV1: the dashboard or data ingestion is down for all customers, or data loss is possible. SEV2: a major feature is broken for many customers, or one large Business customer is fully affected. SEV3: a minor feature is degraded and a workaround exists.

## On-call rotation

Two engineers are on call each week, a primary and a secondary, with handover every Monday at 10:00. The primary must acknowledge a page within 5 minutes; if not, the page escalates to the secondary and then to the engineering manager.

## Declaring an incident

Anyone can declare an incident by typing /incident in Slack. This opens a channel, starts a timeline and pages the on-call engineer. For SEV1 the incident commander posts a status page update within 15 minutes.

## Postmortems

Every SEV1 and SEV2 needs a blameless postmortem within five working days. It lists the timeline, root cause, impact and follow-up actions with owners. Postmortems are reviewed in the weekly reliability meeting.
