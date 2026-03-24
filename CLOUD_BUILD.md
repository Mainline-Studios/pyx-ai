# Cloud Build (pyxaiapi)

## Status code 51 / "untrusted builder"

If the build fails with:

`failed to build: executing lifecycle. This may be the result of using an untrusted builder: failed with status code: 51`

the pack step is not trusting the builder. **Fix:** add `--trust-builder` to the pack build command.

## Using this repo's config

This repo includes a **cloudbuild.yaml** that runs pack with `--trust-builder` and `--builder=gcr.io/buildpacks/builder:google-24`. Use it by running from the repo root:

```bash
gcloud builds submit .
```

The image is built and pushed to `us-central1-docker.pkg.dev/PROJECT_ID/cloud-run-source-deploy/pyxaiapi` by default. Override with substitutions:

```bash
gcloud builds submit . --substitutions=_REGION=us-west2,_REPO=my-repo,_NAME=pyxaiapi
```

## If your trigger uses a different config

In your Cloud Build trigger or step that runs pack, add **`--trust-builder`** to the pack args, for example:

```yaml
args:
  - build
  - YOUR_IMAGE_URI
  - --builder=gcr.io/buildpacks/builder:google-24
  - --trust-builder
  - --network=cloudbuild
```

Without `--trust-builder`, the lifecycle runs in untrusted mode and can fail with status 51.
