# Free-Tier Deployment Decision

**Checked:** 2026-10-11. Azure deployment is intentionally skipped because the user has no Azure credits.

## Selected target: Render Free web service

Render offers free web-service instances for hobby and preview use, but explicitly says not to use them for production. A free web service uses 0.1 CPU and 512 MB RAM, spins down after 15 minutes without inbound traffic, and takes about one minute to spin up on the next request. Free instances have 750 instance-hours per workspace each month, may restart at any time, cannot use persistent disks, and cannot scale beyond one instance. This fits a resume/demo endpoint if its cold-start, availability, and resource constraints are stated clearly; it is not a production/SLA claim. [1]

Render supports prebuilt Docker images from GitHub Container Registry. A private GHCR image requires a registry credential with `read:packages` access. Image-backed services do not automatically redeploy when a tag changes, but a service deploy hook can redeploy a specific image tag or digest through its URL-encoded `imgURL` query parameter. The path/registry must match the image configured on the Render service. [2] [3]

Render documents a health-gated zero-downtime deployment sequence for web services: the old instance continues receiving traffic while the new one starts, routing moves after the new instance is healthy, and the prior instance receives `SIGTERM` 60 seconds later. An HTTP health-check path can gate readiness. A persistent disk disables the zero-downtime sequence, so this demo should not attach one. These are provider-documented behaviors, not a measured availability guarantee for this particular free instance. [4] [5]

## Why not Fly.io

Fly.io's current official pricing page says there is no free tier or monthly free allowance for new organizations. Its trial is limited to two machine-runtime hours or seven days, whichever comes first; an always-on smallest machine is billed at $2.19 per 30 days before outbound data charges. Because the requested target is a no-cost free-tier alternative, Fly.io is not selected. [6]

## Implementation implications

- Configure the Render web service to use the GHCR image `ghcr.io/<owner>/fraud-detection-platform:latest`, set its HTTP health check to `/health`, and use the Free compute plan without a persistent disk.
- Configure a Render GHCR read credential in Render if the package is private. Store the Render deploy hook URL as the GitHub Actions secret `RENDER_DEPLOY_HOOK_URL`.
- The workflow should deploy the immutable commit-SHA image by URL-encoding it in the deploy hook's `imgURL` parameter. Retain the `latest` tag for the service's default image reference and for simple manual recovery.
- The Docker image now uses Render's injected `PORT` (tested with `PORT=10000`) and defaults to one Uvicorn worker to limit memory. Under a local Docker simulation capped at 512 MB and 0.1 CPU, `/health` and `/predict` succeeded, the Docker health check became healthy, and the container was not OOM-killed. The loaded container used about 248.8 MiB of 512 MiB.
- In a 30-request sequential benchmark under that same local 0.1-CPU cap, median latency was **203.8 ms**, p95 **297.5 ms**, and max **299.0 ms**. This simulation did **not** meet the sub-100 ms median target. It ran on the sandbox host with a CPU quota and is not a measurement from Render; do not describe it as Render performance. Measure again after deployment and report the actual deployed median and tail separately. The free plan's cold start after idle is an additional delay not represented by this warm-request benchmark.
- A separate 250-request / concurrency-8 benchmark of the final one-worker container on the full 6-logical-CPU sandbox measured **118.8 ms median**, **239.9 ms p95**, and **331.7 ms max**. It also missed the sub-100 ms median target. This differs from the earlier two-worker API benchmark and should be reported separately.
- The public demo will spin down when idle, so its cold-start latency will differ from the local benchmark. Re-benchmark after deployment and report target-specific median and tail latency separately.
- Until a real recent feature window is supplied through `CURRENT_WINDOW_URL`, the drift job may evaluate the historical holdout proxy for demonstration, but that proxy must not trigger automatic retraining.

## References

[1]: https://render.com/docs/free "Render: Deploy for Free"
[2]: https://render.com/docs/deploying-an-image "Render: Deploy a Prebuilt Docker Image"
[3]: https://render.com/docs/deploy-hooks "Render: Deploy Hooks"
[4]: https://render.com/docs/deploys "Render: Deploying on Render"
[5]: https://render.com/docs/health-checks "Render: Health Checks"
[6]: https://docs.fly.io/about/pricing "Fly.io: Pricing"
