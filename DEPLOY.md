# Deploying PocketSmart to Cloud Run (asia-south1, Mumbai)

These are one-time setup steps, followed by one deploy command. Run them in **PowerShell** from the project folder. Cloud Build builds the image, so you don't need Docker installed.

## 0. Before you deploy
```powershell
.venv\Scripts\python -m src.evaluate --report   # results/metrics.json must be current...
git status                                     # ...and committed: the build verifies the model against it
```

## 1. Install and sign in (one time)
```powershell
winget install --id Google.CloudSDK            # or the installer from cloud.google.com/sdk/docs/install
# close and reopen PowerShell, then:
gcloud auth login
gcloud config set project YOUR_PROJECT_ID      # a project with billing enabled
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## 2. Deploy
```powershell
gcloud run deploy pocketsmart --source . --region asia-south1 --allow-unauthenticated `
  --memory 1Gi --cpu 1 --timeout 60 --max-instances 3 `
  --set-env-vars "GEMINI_MODEL=gemini-3.5-flash-lite"
```
- The first time, it may ask to create an Artifact Registry repository. Answer **Y**.
- In the build log, look for **`BUILD MODEL MATCHES COMMITTED METRICS`**. If the rebuilt model doesn't reproduce `results/metrics.json`, the build fails on purpose.
- At the end, the command prints the **Service URL**. That's your live URL.

## 3. Add the Gemini key (never in git, never in the image)
`Read-Host` keeps the key out of your PowerShell history:
```powershell
$k = Read-Host "Paste GEMINI_API_KEY"
gcloud run services update pocketsmart --region asia-south1 --update-env-vars "GEMINI_API_KEY=$k"
Remove-Variable k
```
Optionally, confirm the model name with the same key: `$env:GEMINI_API_KEY=...; .venv\Scripts\python -m src.gemini --list-models`.
If the model isn't found, redeploy with `--update-env-vars GEMINI_MODEL=<a name from the list>`.

## 4. Verify
```powershell
$u = gcloud run services describe pocketsmart --region asia-south1 --format "value(status.url)"
curl.exe "$u/health"     # metrics_verified: true, model_sha256
curl.exe "$u/metrics"    # the same numbers as the README
start $u                 # open on the laptop; then open the same URL on your phone
```

## 5. Demo day
Cold starts on mobile data are slow. Keep one instance warm for the demo, then turn it off:
```powershell
gcloud run services update pocketsmart --region asia-south1 --min-instances 1   # before the demo
gcloud run services update pocketsmart --region asia-south1 --min-instances 0   # after (stops billing)
```

## Redeploy after the real labels are ingested
```powershell
.venv\Scripts\python -m data.ingest_labels
.venv\Scripts\python -m src.evaluate --report
git add results README.md; git commit -m "Real validation results"
gcloud run deploy pocketsmart --source . --region asia-south1   # env vars are kept
```

## If the deploy fails on the day
Run it locally and demo from the laptop's Wi-Fi hotspot. `/advice` works offline via its fallback:
```powershell
.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
